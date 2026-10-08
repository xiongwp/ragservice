// Package parse turns uploaded files into Markdown.
//
// Plain formats (txt, md, html) are handled in-process. Layout formats (pdf,
// docx, pptx, xlsx) go to a docling-serve sidecar: Go has nothing comparable
// for tables and reading order, and keeping it out of process isolates its
// memory use and crashes from the service.
package parse

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/simplifiedchinese"
)

// ErrUnsupported marks input that will never parse; retrying is pointless.
var ErrUnsupported = errors.New("unsupported document")

var formats = map[string]string{
	".txt": "txt", ".text": "txt", ".log": "txt", ".csv": "txt",
	".md": "md", ".markdown": "md",
	".html": "html", ".htm": "html",
	".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx",
}

// FormatOf maps a file name to a format, or "" if unsupported.
func FormatOf(name string) string {
	return formats[strings.ToLower(filepath.Ext(name))]
}

func needsDocling(format string) bool {
	switch format {
	case "pdf", "docx", "pptx", "xlsx":
		return true
	}
	return false
}

type Parser struct {
	doclingURL string
	http       *http.Client  // per request; long conversions are polled, not held open
	timeout    time.Duration // whole conversion, including the wait for a slot
	slots      chan struct{} // docling is CPU-bound: more parallel jobs only make each slower

	// Encoding for legacy text files whose encoding cannot be detected
	// with confidence (short files, unsupported scripts). Default GB18030.
	fallback     encoding.Encoding
	fallbackName string
}

// SetFallbackEncoding changes the encoding assumed for undetectable legacy
// text files (e.g. "shift_jis" for a Japanese deployment).
func (p *Parser) SetFallbackEncoding(label string) error {
	enc, name, ok := FallbackEncoding(label)
	if !ok {
		return fmt.Errorf("unknown text encoding %q", label)
	}
	p.fallback, p.fallbackName = enc, name
	return nil
}

// New returns a parser. timeout bounds one document's conversion end to end;
// concurrency caps how many documents are sent to docling at once.
func New(doclingURL string, timeout time.Duration, concurrency int) *Parser {
	return &Parser{doclingURL: doclingURL, http: &http.Client{Timeout: 2 * time.Minute},
		timeout: timeout, slots: make(chan struct{}, max(1, concurrency)),
		fallback: simplifiedchinese.GB18030, fallbackName: "gb18030"}
}

// Parse returns the document as Markdown.
func (p *Parser) Parse(ctx context.Context, name, format string, data []byte) (string, error) {
	switch {
	case format == "txt" || format == "md":
		text, _ := decodeText(data, p.fallback, p.fallbackName)
		return text, nil
	case format == "html":
		page, _ := decodeHTML(data, p.fallback, p.fallbackName)
		md, err := htmltomarkdown.ConvertString(page)
		if err != nil {
			return "", fmt.Errorf("%w: html: %v", ErrUnsupported, err)
		}
		return md, nil
	case needsDocling(format):
		if p.doclingURL == "" {
			return "", fmt.Errorf("%w: %s needs the docling sidecar but DOCLING_URL is not set", ErrUnsupported, format)
		}
		return p.docling(ctx, name, data)
	}
	return "", fmt.Errorf("%w: format %q", ErrUnsupported, format)
}

type doclingResponse struct {
	Status   string `json:"status"`
	Errors   []any  `json:"errors"`
	Document struct {
		MDContent string `json:"md_content"`
	} `json:"document"`
}

func (p *Parser) docling(ctx context.Context, name string, data []byte) (string, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	for k, v := range map[string]string{
		"to_formats":        "md",
		"image_export_mode": "placeholder",
		"do_ocr":            "true",
		"table_mode":        "accurate",
	} {
		if err := w.WriteField(k, v); err != nil {
			return "", err
		}
	}
	fw, err := w.CreateFormFile("files", filepath.Base(name))
	if err != nil {
		return "", err
	}
	if _, err := fw.Write(data); err != nil {
		return "", err
	}
	if err := w.Close(); err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	select {
	case p.slots <- struct{}{}:
		defer func() { <-p.slots }()
	case <-ctx.Done():
		return "", fmt.Errorf("docling: waiting for a free slot: %w", ctx.Err())
	}

	// Submit asynchronously and poll: a 100-page PDF can take many minutes
	// on CPU, longer than any single HTTP request should stay open.
	var task struct {
		TaskID string `json:"task_id"`
	}
	if err := p.doclingCall(ctx, http.MethodPost, "/v1/convert/file/async", &body, w.FormDataContentType(), &task); err != nil {
		return "", err
	}
	for {
		var st struct {
			TaskStatus   string `json:"task_status"`
			ErrorMessage string `json:"error_message"`
		}
		if err := p.doclingCall(ctx, http.MethodGet, "/v1/status/poll/"+task.TaskID+"?wait=10", nil, "", &st); err != nil {
			return "", err
		}
		if st.TaskStatus == "success" || st.TaskStatus == "partial_success" {
			break
		}
		if st.TaskStatus == "failure" {
			return "", fmt.Errorf("%w: docling could not convert %s: %s", ErrUnsupported, name, st.ErrorMessage)
		}
		select {
		case <-time.After(2 * time.Second):
		case <-ctx.Done():
			return "", fmt.Errorf("docling: conversion still running after %s: %w", p.timeout, ctx.Err())
		}
	}
	var out doclingResponse
	if err := p.doclingCall(ctx, http.MethodGet, "/v1/result/"+task.TaskID, nil, "", &out); err != nil {
		return "", err
	}
	if out.Status != "success" && out.Status != "partial_success" {
		return "", fmt.Errorf("docling: status %q: %v", out.Status, out.Errors)
	}
	// Images carry no text for retrieval; drop the placeholders docling leaves.
	return strings.ReplaceAll(out.Document.MDContent, "<!-- image -->", ""), nil
}

func (p *Parser) doclingCall(ctx context.Context, method, path string, body io.Reader, contentType string, out any) error {
	req, err := http.NewRequestWithContext(ctx, method, p.doclingURL+path, body)
	if err != nil {
		return err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := p.http.Do(req)
	if err != nil {
		return fmt.Errorf("docling: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("docling: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("docling: %s %s: HTTP %d: %s", method, path, resp.StatusCode, truncate(raw, 500))
		if resp.StatusCode == http.StatusUnprocessableEntity || resp.StatusCode == http.StatusUnsupportedMediaType {
			return fmt.Errorf("%w: %v", ErrUnsupported, err)
		}
		return err
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("docling: decode response: %w", err)
	}
	return nil
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…"
}
