package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

type uploaded struct {
	name, content string
	metadata      map[string]any
}

// fakes serves a Dify knowledge base with four documents and a ragd that
// records uploads.
func fakes(t *testing.T) (dify, rag *httptest.Server, uploads *[]uploaded) {
	var mu sync.Mutex
	var got []uploaded
	seen := map[string]bool{}
	dify = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer dataset-key" && !strings.HasPrefix(r.URL.Path, "/files/") {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		switch {
		case r.URL.Path == "/v1/datasets/kb/documents" && r.URL.Query().Get("page") == "1":
			w.Write([]byte(`{"has_more": true, "data": [
				{"id": "d1", "name": "手册.pdf", "data_source_type": "upload_file", "indexing_status": "completed", "enabled": true,
				 "doc_metadata": [{"name": "dept", "value": "hr"}, {"name": "empty", "value": null}]},
				{"id": "d2", "name": "notion 页面", "data_source_type": "notion_import", "indexing_status": "completed", "enabled": true}]}`))
		case r.URL.Path == "/v1/datasets/kb/documents" && r.URL.Query().Get("page") == "2":
			w.Write([]byte(`{"has_more": false, "data": [
				{"id": "d3", "name": "处理中.txt", "data_source_type": "upload_file", "indexing_status": "indexing", "enabled": true},
				{"id": "d4", "name": "已禁用.txt", "data_source_type": "upload_file", "indexing_status": "completed", "enabled": false}]}`))
		case r.URL.Path == "/v1/datasets/kb/documents/d1/download":
			w.Write([]byte(`{"url": "/files/d1?sign=abc"}`)) // relative, as Dify returns without FILES_URL
		case r.URL.Path == "/files/d1":
			w.Write([]byte("%PDF-1.7 original bytes"))
		case r.URL.Path == "/v1/datasets/kb/documents/d2/segments":
			if r.URL.Query().Get("page") == "1" {
				w.Write([]byte(`{"has_more": true, "data": [{"position": 2, "content": "第二段", "enabled": true},
					{"position": 1, "content": "第一段", "enabled": true}]}`))
			} else {
				w.Write([]byte(`{"has_more": false, "data": [{"position": 3, "content": "停用段", "enabled": false},
					{"position": 4, "content": "问题", "answer": "答案", "enabled": true}]}`))
			}
		default:
			http.NotFound(w, r)
		}
	}))
	rag = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer rag-key" {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		switch r.URL.Path {
		case "/v1/datasets":
			w.WriteHeader(http.StatusCreated)
			w.Write([]byte(`{"id": "new-ds"}`))
		case "/v1/datasets/new-ds/documents":
			f, h, err := r.FormFile("file")
			if err != nil {
				t.Errorf("upload: %v", err)
				return
			}
			b, _ := io.ReadAll(f)
			var md map[string]any
			json.Unmarshal([]byte(r.FormValue("metadata")), &md)
			mu.Lock()
			dup := seen[string(b)]
			seen[string(b)] = true
			got = append(got, uploaded{h.Filename, string(b), md})
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]any{"duplicate": dup})
		default:
			http.NotFound(w, r)
		}
	}))
	return dify, rag, &got
}

func TestImport(t *testing.T) {
	dify, rag, uploads := fakes(t)
	defer dify.Close()
	defer rag.Close()
	im := &importer{cfg: config{difyURL: dify.URL + "/v1", difyKey: "dataset-key", difyDataset: "kb",
		ragURL: rag.URL, ragKey: "rag-key", create: "导入"}, http: dify.Client(), log: io.Discard}

	sum, err := im.run()
	if err != nil {
		t.Fatal(err)
	}
	if sum.files != 1 || sum.segments != 1 || sum.skipped != 2 || sum.failed != 0 {
		t.Fatalf("summary = %+v", sum)
	}
	byName := map[string]uploaded{}
	for _, u := range *uploads {
		byName[u.name] = u
	}
	pdf := byName["手册.pdf"]
	if pdf.content != "%PDF-1.7 original bytes" || pdf.metadata["dept"] != "hr" || pdf.metadata["dify_document_id"] != "d1" {
		t.Errorf("pdf upload = %+v", pdf)
	}
	if _, ok := pdf.metadata["empty"]; ok {
		t.Errorf("null metadata values should be dropped")
	}
	notion := byName["notion 页面.md"]
	if notion.content != "第一段\n\n第二段\n\nQ: 问题\nA: 答案" {
		t.Errorf("segments joined as %q", notion.content)
	}

	// A second run uploads the same content and ragd reports duplicates.
	im.cfg.create, im.cfg.ragDataset = "", "new-ds"
	sum, _ = im.run()
	if sum.duplicates != 2 || sum.files+sum.segments != 0 {
		t.Errorf("re-run summary = %+v", sum)
	}
}

func TestSegmentsOnlyAndDryRun(t *testing.T) {
	dify, rag, uploads := fakes(t)
	defer dify.Close()
	defer rag.Close()
	im := &importer{cfg: config{difyURL: dify.URL + "/v1", difyKey: "dataset-key", difyDataset: "kb",
		ragURL: rag.URL, ragKey: "rag-key", ragDataset: "new-ds", dryRun: true}, http: dify.Client(), log: io.Discard}
	if _, err := im.run(); err != nil {
		t.Fatal(err)
	}
	if len(*uploads) != 0 {
		t.Errorf("dry run uploaded %d files", len(*uploads))
	}

	name, _, how, err := (&importer{cfg: config{difyURL: dify.URL + "/v1", difyKey: "dataset-key", difyDataset: "kb", segmentsOnly: true},
		http: dify.Client(), log: io.Discard}).content(difyDoc{ID: "d2", Name: "x.pdf", DataSourceType: "upload_file"})
	if err != nil || how != "segments" || name != "x.md" {
		t.Errorf("segments-only: %s %s %v", name, how, err)
	}
}
