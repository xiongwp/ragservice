// difyimport copies the documents of a Dify knowledge base into ragd.
//
//	difyimport -dify-url http://localhost/v1 -dify-key dataset-xxx -dify-dataset <id> \
//	  -rag-url http://localhost:8080 -rag-key <editor key> -create "公司制度"
//
// It talks to both services over their HTTP APIs only (Dify's knowledge-base
// service API, ragd's /v1), so it works whatever database Dify runs on.
//
// For each completed, enabled Dify document it uploads, in order of preference:
//  1. the original file (Dify's download URL), re-parsed by ragd's pipeline;
//  2. otherwise its segments, joined in order into one Markdown file (for
//     text, Notion or web sources, unsupported formats, or -segments-only).
//
// Dify's document metadata becomes ragd metadata, plus source=dify and
// dify_document_id. Re-running is safe: ragd skips identical content.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"rag/internal/parse"
)

type config struct {
	difyURL, difyKey, difyDataset string
	ragURL, ragKey, ragDataset    string
	create                        string
	segmentsOnly, dryRun          bool
}

type difyDoc struct {
	ID             string `json:"id"`
	Name           string `json:"name"`
	DataSourceType string `json:"data_source_type"`
	IndexingStatus string `json:"indexing_status"`
	Enabled        *bool  `json:"enabled"`
	Archived       bool   `json:"archived"`
	DocMetadata    []struct {
		Name  string `json:"name"`
		Value any    `json:"value"`
	} `json:"doc_metadata"`
}

type segment struct {
	Position int    `json:"position"`
	Content  string `json:"content"`
	Answer   string `json:"answer"`
	Enabled  bool   `json:"enabled"`
}

type importer struct {
	cfg  config
	http *http.Client
	log  io.Writer
}

func main() {
	var c config
	flag.StringVar(&c.difyURL, "dify-url", "", "Dify service API base, e.g. http://localhost/v1")
	flag.StringVar(&c.difyKey, "dify-key", "", "Dify knowledge-base API key (dataset-...)")
	flag.StringVar(&c.difyDataset, "dify-dataset", "", "Dify knowledge base id")
	flag.StringVar(&c.ragURL, "rag-url", "http://localhost:8080", "ragd base URL")
	flag.StringVar(&c.ragKey, "rag-key", "", "ragd API key with the editor role")
	flag.StringVar(&c.ragDataset, "rag-dataset", "", "import into this existing ragd dataset")
	flag.StringVar(&c.create, "create", "", "or create a new ragd dataset with this name")
	flag.BoolVar(&c.segmentsOnly, "segments-only", false, "always rebuild documents from Dify's segments")
	flag.BoolVar(&c.dryRun, "dry-run", false, "list what would be imported, change nothing")
	flag.Parse()
	if c.difyURL == "" || c.difyKey == "" || c.difyDataset == "" || c.ragKey == "" || (c.ragDataset == "") == (c.create == "") {
		fmt.Fprintln(os.Stderr, "need -dify-url -dify-key -dify-dataset -rag-key and exactly one of -rag-dataset / -create")
		flag.Usage()
		os.Exit(2)
	}
	c.difyURL, c.ragURL = strings.TrimRight(c.difyURL, "/"), strings.TrimRight(c.ragURL, "/")
	im := &importer{cfg: c, http: &http.Client{Timeout: 5 * time.Minute}, log: os.Stderr}
	sum, err := im.run()
	fmt.Fprintf(os.Stderr, "\nimported %d (original file %d, from segments %d) · already present %d · skipped %d · failed %d\n",
		sum.files+sum.segments, sum.files, sum.segments, sum.duplicates, sum.skipped, sum.failed)
	if err != nil {
		fmt.Fprintln(os.Stderr, "difyimport:", err)
		os.Exit(1)
	}
	if sum.failed > 0 {
		os.Exit(1)
	}
}

type summary struct{ files, segments, duplicates, skipped, failed int }

func (im *importer) run() (summary, error) {
	var sum summary
	docs, err := im.listDocuments()
	if err != nil {
		return sum, fmt.Errorf("list Dify documents: %w", err)
	}
	fmt.Fprintf(im.log, "Dify knowledge base has %d documents\n", len(docs))

	target := im.cfg.ragDataset
	if im.cfg.create != "" && !im.cfg.dryRun {
		var ds struct {
			ID string `json:"id"`
		}
		if err := im.ragJSON("POST", "/v1/datasets", map[string]any{"name": im.cfg.create,
			"description": "imported from Dify knowledge base " + im.cfg.difyDataset}, &ds); err != nil {
			return sum, fmt.Errorf("create ragd dataset: %w", err)
		}
		target = ds.ID
		fmt.Fprintf(im.log, "created ragd dataset %s (use this id as the external knowledge id in Dify)\n", target)
	}

	for _, d := range docs {
		if d.IndexingStatus != "completed" || d.Archived || (d.Enabled != nil && !*d.Enabled) {
			fmt.Fprintf(im.log, "skip   %s (status %s, enabled %v, archived %v)\n", d.Name, d.IndexingStatus, d.Enabled == nil || *d.Enabled, d.Archived)
			sum.skipped++
			continue
		}
		metadata := map[string]any{"source": "dify", "dify_document_id": d.ID}
		for _, m := range d.DocMetadata {
			if m.Name != "" && m.Value != nil {
				metadata[m.Name] = m.Value
			}
		}

		name, data, how, err := im.content(d)
		if err != nil {
			fmt.Fprintf(im.log, "FAIL   %s: %v\n", d.Name, err)
			sum.failed++
			continue
		}
		if im.cfg.dryRun {
			fmt.Fprintf(im.log, "would  %s → %s (%s, %d bytes)\n", d.Name, name, how, len(data))
			continue
		}
		dup, err := im.upload(target, name, data, metadata)
		switch {
		case err != nil:
			fmt.Fprintf(im.log, "FAIL   %s: %v\n", d.Name, err)
			sum.failed++
		case dup:
			fmt.Fprintf(im.log, "exists %s\n", d.Name)
			sum.duplicates++
		default:
			fmt.Fprintf(im.log, "ok     %s (%s)\n", d.Name, how)
			if how == "original file" {
				sum.files++
			} else {
				sum.segments++
			}
		}
	}
	return sum, nil
}

// content returns the file to upload for a Dify document.
func (im *importer) content(d difyDoc) (name string, data []byte, how string, err error) {
	if !im.cfg.segmentsOnly && d.DataSourceType == "upload_file" && parse.FormatOf(d.Name) != "" {
		data, err := im.download(d.ID)
		if err == nil && len(data) > 0 {
			return d.Name, data, "original file", nil
		}
		fmt.Fprintf(im.log, "note   %s: original file unavailable (%v); using segments\n", d.Name, err)
	}
	segs, err := im.listSegments(d.ID)
	if err != nil {
		return "", nil, "", fmt.Errorf("segments: %w", err)
	}
	text := joinSegments(segs)
	if strings.TrimSpace(text) == "" {
		return "", nil, "", fmt.Errorf("document has no enabled segments")
	}
	return strings.TrimSuffix(d.Name, filepath.Ext(d.Name)) + ".md", []byte(text), "segments", nil
}

func joinSegments(segs []segment) string {
	sort.SliceStable(segs, func(i, j int) bool { return segs[i].Position < segs[j].Position })
	var parts []string
	for _, s := range segs {
		if !s.Enabled {
			continue
		}
		if s.Answer != "" { // Q&A-mode knowledge bases
			parts = append(parts, "Q: "+strings.TrimSpace(s.Content)+"\nA: "+strings.TrimSpace(s.Answer))
		} else {
			parts = append(parts, strings.TrimSpace(s.Content))
		}
	}
	return strings.Join(parts, "\n\n")
}

// ---- Dify ----

func (im *importer) difyGet(path string, out any) error {
	req, err := http.NewRequest("GET", im.cfg.difyURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+im.cfg.difyKey)
	return im.doJSON(req, out)
}

func (im *importer) listDocuments() ([]difyDoc, error) {
	var all []difyDoc
	for page := 1; ; page++ {
		var res struct {
			Data    []difyDoc `json:"data"`
			HasMore bool      `json:"has_more"`
		}
		if err := im.difyGet(fmt.Sprintf("/datasets/%s/documents?page=%d&limit=100", url.PathEscape(im.cfg.difyDataset), page), &res); err != nil {
			return nil, err
		}
		all = append(all, res.Data...)
		if !res.HasMore || len(res.Data) == 0 {
			return all, nil
		}
	}
}

func (im *importer) listSegments(docID string) ([]segment, error) {
	var all []segment
	for page := 1; ; page++ {
		var res struct {
			Data    []segment `json:"data"`
			HasMore bool      `json:"has_more"`
		}
		path := fmt.Sprintf("/datasets/%s/documents/%s/segments?page=%d&limit=100",
			url.PathEscape(im.cfg.difyDataset), url.PathEscape(docID), page)
		if err := im.difyGet(path, &res); err != nil {
			return nil, err
		}
		all = append(all, res.Data...)
		if !res.HasMore || len(res.Data) == 0 {
			return all, nil
		}
	}
}

func (im *importer) download(docID string) ([]byte, error) {
	var res struct {
		URL string `json:"url"`
	}
	if err := im.difyGet(fmt.Sprintf("/datasets/%s/documents/%s/download",
		url.PathEscape(im.cfg.difyDataset), url.PathEscape(docID)), &res); err != nil {
		return nil, err
	}
	fileURL, err := resolve(im.cfg.difyURL, res.URL)
	if err != nil {
		return nil, err
	}
	resp, err := im.http.Get(fileURL) // a signed URL: no API key needed
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download: HTTP %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// resolve turns a possibly relative download URL ("/files/...") into an
// absolute one on the Dify host.
func resolve(base, ref string) (string, error) {
	b, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	r, err := url.Parse(ref)
	if err != nil {
		return "", err
	}
	return b.ResolveReference(r).String(), nil
}

// ---- ragd ----

func (im *importer) ragJSON(method, path string, body, out any) error {
	b, _ := json.Marshal(body)
	req, err := http.NewRequest(method, im.cfg.ragURL+path, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+im.cfg.ragKey)
	req.Header.Set("Content-Type", "application/json")
	return im.doJSON(req, out)
}

func (im *importer) upload(datasetID, name string, data []byte, metadata map[string]any) (duplicate bool, err error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile("file", name)
	if err != nil {
		return false, err
	}
	fw.Write(data)
	md, _ := json.Marshal(metadata)
	w.WriteField("metadata", string(md))
	w.Close()
	req, err := http.NewRequest("POST", im.cfg.ragURL+"/v1/datasets/"+url.PathEscape(datasetID)+"/documents", &buf)
	if err != nil {
		return false, err
	}
	req.Header.Set("Authorization", "Bearer "+im.cfg.ragKey)
	req.Header.Set("Content-Type", w.FormDataContentType())
	var res struct {
		Duplicate bool `json:"duplicate"`
	}
	err = im.doJSON(req, &res)
	return res.Duplicate, err
}

func (im *importer) doJSON(req *http.Request, out any) error {
	resp, err := im.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("%s %s: HTTP %d: %.300s", req.Method, req.URL.Path, resp.StatusCode, raw)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(raw, out)
}
