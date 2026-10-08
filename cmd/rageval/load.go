package main

// rageval load: put the same folder of documents into ragd and/or Dify and
// wait until both have indexed it, so the comparison runs on identical data.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"rag/internal/parse"
)

func runLoad(args []string) {
	fs := flag.NewFlagSet("load", flag.ExitOnError)
	docs := fs.String("docs", "", "folder of documents")
	name := fs.String("name", "rageval", "knowledge base name to create")
	ragURL := fs.String("rag-url", "", "ragd base URL (empty: skip ragd)")
	ragKey := fs.String("rag-key", "", "ragd key with the editor role")
	difyURL := fs.String("dify-url", "", "Dify service API base, e.g. http://localhost/v1 (empty: skip Dify)")
	difyKey := fs.String("dify-key", "", "Dify knowledge API key (dataset-...)")
	embedModel := fs.String("dify-embedding-model", "BAAI/bge-m3", "embedding model Dify should use")
	embedProvider := fs.String("dify-embedding-provider", "langgenius/siliconflow/siliconflow", "its provider in Dify")
	fs.Parse(args)
	if *docs == "" || (*ragURL == "" && *difyURL == "") {
		fatal(fmt.Errorf("load needs -docs and at least one of -rag-url / -dify-url"))
	}
	files, err := listFiles(*docs)
	if err != nil {
		fatal(err)
	}
	fmt.Fprintf(os.Stderr, "%d files\n", len(files))
	c := &http.Client{Timeout: 10 * time.Minute}

	if *ragURL != "" {
		id, err := loadRagd(c, strings.TrimRight(*ragURL, "/"), *ragKey, *name, files)
		if err != nil {
			fatal(fmt.Errorf("ragd: %w", err))
		}
		fmt.Printf("ragd knowledge id: %s\n", id)
	}
	if *difyURL != "" {
		id, err := loadDify(c, strings.TrimRight(*difyURL, "/"), *difyKey, *name, *embedModel, *embedProvider, files)
		if err != nil {
			fatal(fmt.Errorf("dify: %w", err))
		}
		fmt.Printf("dify knowledge id: %s\n", id)
	}
}

func listFiles(dir string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && !strings.HasPrefix(d.Name(), ".") && parse.FormatOf(d.Name()) != "" {
			out = append(out, path)
		}
		return err
	})
	return out, err
}

func call(c *http.Client, method, url, key string, body io.Reader, contentType string, out any) error {
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("%s %s: HTTP %d: %.400s", method, req.URL.Path, resp.StatusCode, raw)
	}
	if out != nil {
		return json.Unmarshal(raw, out)
	}
	return nil
}

func callJSON(c *http.Client, method, url, key string, body, out any) error {
	b, _ := json.Marshal(body)
	return call(c, method, url, key, bytes.NewReader(b), "application/json", out)
}

// multipartFile builds a form with the file plus extra text fields.
func multipartFile(path string, fields map[string]string) (*bytes.Buffer, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for k, v := range fields {
		w.WriteField(k, v)
	}
	fw, err := w.CreateFormFile("file", filepath.Base(path))
	if err != nil {
		return nil, "", err
	}
	fw.Write(data)
	w.Close()
	return &buf, w.FormDataContentType(), nil
}

func loadRagd(c *http.Client, base, key, name string, files []string) (string, error) {
	var ds struct {
		ID string `json:"id"`
	}
	if err := callJSON(c, "POST", base+"/v1/datasets", key, map[string]any{"name": name}, &ds); err != nil {
		return "", err
	}
	for _, f := range files {
		body, ct, err := multipartFile(f, nil)
		if err != nil {
			return "", err
		}
		if err := call(c, "POST", base+"/v1/datasets/"+ds.ID+"/documents", key, body, ct, nil); err != nil {
			fmt.Fprintf(os.Stderr, "ragd upload %s: %v\n", filepath.Base(f), err)
		}
	}
	return ds.ID, waitIndexed("ragd", func() (done, failed, total int, err error) {
		var res struct {
			Data []struct {
				Name, Status, Error string
			} `json:"data"`
		}
		if err := callJSON(c, "GET", base+"/v1/datasets/"+ds.ID+"/documents", key, nil, &res); err != nil {
			return 0, 0, 0, err
		}
		for _, d := range res.Data {
			switch d.Status {
			case "completed":
				done++
			case "failed":
				failed++
				fmt.Fprintf(os.Stderr, "ragd failed %s: %s\n", d.Name, d.Error)
			}
		}
		return done, failed, len(res.Data), nil
	})
}

func loadDify(c *http.Client, base, key, name, embedModel, embedProvider string, files []string) (string, error) {
	var ds struct {
		ID string `json:"id"`
	}
	err := callJSON(c, "POST", base+"/datasets", key, map[string]any{
		"name": name, "indexing_technique": "high_quality",
		"embedding_model": embedModel, "embedding_model_provider": embedProvider,
		"retrieval_model": map[string]any{"search_method": "semantic_search", "reranking_enable": false,
			"top_k": 10, "score_threshold_enabled": false},
	}, &ds)
	if err != nil {
		return "", err
	}
	// Dify's own defaults for chunking ("automatic"), as a user would get them.
	data, _ := json.Marshal(map[string]any{
		"indexing_technique": "high_quality", "doc_form": "text_model", "doc_language": "Chinese",
		"process_rule": map[string]any{"mode": "automatic"},
	})
	for _, f := range files {
		body, ct, err := multipartFile(f, map[string]string{"data": string(data)})
		if err != nil {
			return "", err
		}
		if err := call(c, "POST", base+"/datasets/"+ds.ID+"/document/create-by-file", key, body, ct, nil); err != nil {
			fmt.Fprintf(os.Stderr, "dify upload %s: %v\n", filepath.Base(f), err)
		}
	}
	return ds.ID, waitIndexed("dify", func() (done, failed, total int, err error) {
		for page := 1; ; page++ {
			var res struct {
				Data []struct {
					Name           string `json:"name"`
					IndexingStatus string `json:"indexing_status"`
					Error          string `json:"error"`
				} `json:"data"`
				HasMore bool `json:"has_more"`
			}
			if err := callJSON(c, "GET", fmt.Sprintf("%s/datasets/%s/documents?page=%d&limit=100", base, ds.ID, page), key, nil, &res); err != nil {
				return 0, 0, 0, err
			}
			for _, d := range res.Data {
				total++
				switch d.IndexingStatus {
				case "completed":
					done++
				case "error", "paused":
					failed++
					fmt.Fprintf(os.Stderr, "dify failed %s: %s\n", d.Name, d.Error)
				}
			}
			if !res.HasMore {
				return done, failed, total, nil
			}
		}
	})
}

func waitIndexed(who string, check func() (done, failed, total int, err error)) error {
	start := time.Now()
	for {
		done, failed, total, err := check()
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s: %d/%d indexed, %d failed (%s)\n", who, done, total, failed, time.Since(start).Round(time.Second))
		if done+failed == total {
			return nil
		}
		if time.Since(start) > 2*time.Hour {
			return fmt.Errorf("still indexing after 2h")
		}
		time.Sleep(10 * time.Second)
	}
}
