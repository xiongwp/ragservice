package api_test

// End-to-end: real MySQL + Qdrant + job queue, fake embedding model.
// Runs only when both env vars are set. The MySQL database is wiped; each
// run uses a fresh Qdrant collection and drops it afterwards.
//
//	make test-deps   # mysql:8.4 on 33306, qdrant on 56333/56334
//	RAG_TEST_MYSQL_DSN='rag:rag@tcp(localhost:33306)/rag_test' RAG_TEST_QDRANT=localhost:56334 go test ./internal/api/

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log/slog"
	"math"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-sql-driver/mysql"
	"github.com/google/uuid"

	"rag/internal/api"
	"rag/internal/auth"
	"rag/internal/chunk"
	"rag/internal/config"
	"rag/internal/contextual"
	"rag/internal/db"
	"rag/internal/embedding"
	"rag/internal/ingest"
	"rag/internal/lexical"
	"rag/internal/llm"
	"rag/internal/parse"
	"rag/internal/queue"
	"rag/internal/rerank"
	"rag/internal/retrieve"
	"rag/internal/rewrite"
	"rag/internal/store"
	"rag/internal/vector"
)

const apiKey = "test-key"

// fakeEmbedder hashes character bigrams into 1024 buckets: texts sharing
// words land close together, which is all retrieval tests need.
func fakeEmbedder(t *testing.T, inputs *atomic.Int64) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("fake embedder: %v", err)
		}
		inputs.Add(int64(len(req.Input)))
		type item struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		data := make([]item, len(req.Input))
		for i, text := range req.Input {
			v := make([]float32, config.EmbeddingDim)
			runes := []rune(strings.ToLower(text))
			for j := 0; j+1 < len(runes); j++ {
				h := fnv.New32a()
				h.Write([]byte(string(runes[j : j+2])))
				v[h.Sum32()%uint32(len(v))]++
			}
			var norm float64
			for _, x := range v {
				norm += float64(x * x)
			}
			if norm == 0 {
				v[0], norm = 1, 1
			}
			for k := range v {
				v[k] /= float32(math.Sqrt(norm))
			}
			data[i] = item{i, v}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
}

// fakeReranker scores a passage by the share of the query's characters it
// contains — crude, but deterministic and query-dependent. While broken is
// set it answers 500.
func fakeReranker(t *testing.T, broken *atomic.Bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if broken.Load() {
			http.Error(w, "down", http.StatusInternalServerError)
			return
		}
		var req struct {
			Query     string   `json:"query"`
			Documents []string `json:"documents"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("fake reranker: %v", err)
		}
		type result struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
		}
		var out []result
		q := []rune(req.Query)
		for i, d := range req.Documents {
			hit := 0
			for _, r := range q {
				if strings.ContainsRune(d, r) {
					hit++
				}
			}
			out = append(out, result{i, float64(hit) / float64(len(q))})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": out})
	}))
}

// fakeLLM answers each prompt type with a fixed text and counts calls.
type fakeLLM struct {
	srv                         *httptest.Server
	contexts, standalone, multi atomic.Int64
	broken                      atomic.Bool
}

func newFakeLLM(t *testing.T) *fakeLLM {
	f := &fakeLLM{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.broken.Load() {
			http.Error(w, "overloaded", http.StatusServiceUnavailable)
			return
		}
		var req struct {
			Messages []struct{ Content string } `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
			t.Errorf("fake LLM: bad request %v", err)
			return
		}
		prompt := req.Messages[0].Content
		var answer string
		switch {
		case strings.Contains(prompt, "<chunk>"):
			f.contexts.Add(1)
			answer = "<think>reasoning to strip</think>本段出自公司员工手册，介绍休假与报销制度。"
		case strings.Contains(prompt, "不依赖对话历史"):
			f.standalone.Add(1)
			answer = "工龄满十年的员工年假有几天"
		case strings.Contains(prompt, "不同的表述"):
			f.multi.Add(1)
			answer = "1. 年休假天数规定\n2. 带薪年假怎么计算\n3. 工龄与年假的关系"
		default:
			answer = "员工工龄满十年的，年假为十天。"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": answer}}}})
	}))
	return f
}

type env struct {
	t   *testing.T
	url string
}

func (e *env) do(method, path string, body io.Reader, contentType string, auth string) (int, map[string]any) {
	e.t.Helper()
	req, _ := http.NewRequest(method, e.url+path, body)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	raw, _ := io.ReadAll(resp.Body)
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &out); err != nil {
			e.t.Fatalf("%s %s: non-JSON response %q", method, path, raw)
		}
	}
	return resp.StatusCode, out
}

func (e *env) json(method, path string, v any) (int, map[string]any) {
	return e.jsonAs(apiKey, method, path, v)
}

func (e *env) jsonAs(key, method, path string, v any) (int, map[string]any) {
	b, _ := json.Marshal(v)
	return e.do(method, path, bytes.NewReader(b), "application/json", "Bearer "+key)
}

func (e *env) upload(datasetID, name, content string, metadata map[string]any) (int, map[string]any) {
	return e.uploadAs(apiKey, datasetID, name, content, metadata)
}

func (e *env) uploadAs(key, datasetID, name, content string, metadata map[string]any) (int, map[string]any) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, _ := w.CreateFormFile("file", name)
	fw.Write([]byte(content))
	if metadata != nil {
		md, _ := json.Marshal(metadata)
		w.WriteField("metadata", string(md))
	}
	w.Close()
	return e.do("POST", "/v1/datasets/"+datasetID+"/documents", &buf, w.FormDataContentType(), "Bearer "+key)
}

func (e *env) waitIndexed(docID string) map[string]any {
	e.t.Helper()
	return e.waitIndexedFor(docID, 30*time.Second)
}

func (e *env) waitIndexedFor(docID string, timeout time.Duration) map[string]any {
	e.t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		_, doc := e.json("GET", "/v1/documents/"+docID, nil)
		switch doc["status"] {
		case "completed":
			return doc
		case "failed":
			e.t.Fatalf("indexing failed: %v", doc["error"])
		}
		time.Sleep(100 * time.Millisecond)
	}
	e.t.Fatalf("document %s not indexed in time", docID)
	return nil
}

func (e *env) retrieve(body map[string]any) []map[string]any {
	e.t.Helper()
	status, out := e.json("POST", "/retrieval", body)
	if status != http.StatusOK {
		e.t.Fatalf("/retrieval: %d %v", status, out)
	}
	var records []map[string]any
	for _, r := range out["records"].([]any) {
		records = append(records, r.(map[string]any))
	}
	return records
}

const handbook = `# 员工手册

## 年假制度

员工每年享有带薪年假。工龄满一年不满十年的，年假为五天；满十年不满二十年的，年假为十天；满二十年的，年假为十五天。年假需提前三个工作日在系统中申请。

## 报销流程

1. 填写报销单并附上发票
2. 部门经理审批
3. 财务复核后五个工作日内打款

| 费用类型 | 标准 |
|---|---|
| 差旅住宿 | 每晚不超过 500 元 |
| 市内交通 | 实报实销 |
`

const itGuide = `<html><body>
<h1>IT 指南</h1>
<h2>VPN 连接</h2>
<p>在家办公时请先连接公司 VPN。VPN 客户端可以在内网门户下载，登录账号与邮箱账号相同。</p>
<h2>密码策略</h2>
<p>系统密码每九十天必须修改一次，长度至少十二位，并包含大小写字母和数字。</p>
</body></html>`

func TestEndToEnd(t *testing.T) {
	dsn, qdrantAddr := os.Getenv("RAG_TEST_MYSQL_DSN"), os.Getenv("RAG_TEST_QDRANT")
	if dsn == "" || qdrantAddr == "" {
		t.Skip("RAG_TEST_MYSQL_DSN / RAG_TEST_QDRANT not set")
	}
	ctx := t.Context()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	// Start from an empty database.
	mcfg, err := mysql.ParseDSN(dsn)
	if err != nil {
		t.Fatal(err)
	}
	wipe, err := db.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{"DROP DATABASE IF EXISTS `" + mcfg.DBName + "`", "CREATE DATABASE `" + mcfg.DBName + "`"} {
		if _, err := wipe.ExecContext(ctx, q); err != nil {
			t.Fatal(err)
		}
	}
	wipe.Close()
	if err := db.Migrate(ctx, dsn, log); err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()

	host, portStr, _ := net.SplitHostPort(qdrantAddr)
	port, _ := strconv.Atoi(portStr)
	vectors, err := vector.Open(ctx, config.QdrantConfig{Host: host, Port: port,
		Collection: fmt.Sprintf("rag_e2e_%d", time.Now().UnixNano())}, config.EmbeddingDim)
	if err != nil {
		t.Fatal(err)
	}
	defer vectors.Close()
	defer vectors.DropCollection(ctx)

	var embedded atomic.Int64
	fake := fakeEmbedder(t, &embedded)
	defer fake.Close()
	fl := newFakeLLM(t)
	defer fl.srv.Close()
	llmCfg := config.LLMConfig{BaseURL: fl.srv.URL, Model: "fake-llm", Timeout: 5 * time.Second, Concurrency: 4}

	cfg := &config.Config{
		APIKeys:        []string{apiKey},
		TenantID:       uuid.MustParse("00000000-0000-0000-0000-000000000001"),
		MaxUploadBytes: 10 << 20,
		Workers:        2,
		Embedding: config.EmbeddingConfig{BaseURL: fake.URL, Model: "fake-bigram", Dim: config.EmbeddingDim,
			BatchSize: 4, Concurrency: 2, Timeout: 5 * time.Second},
		Chunk: config.ChunkDefaults{ParentTokens: 300, ChildTokens: 80},
		LLM:   llmCfg,
	}
	tok, err := chunk.NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	lex, err := lexical.New()
	if err != nil {
		t.Fatal(err)
	}
	var rerankBroken atomic.Bool
	rr := fakeReranker(t, &rerankBroken)
	defer rr.Close()
	st := store.New(sqlDB)
	client := embedding.NewClient(cfg.Embedding)
	q := queue.New(sqlDB, 2, log)
	q.PollInterval = 50 * time.Millisecond
	doclingURL := os.Getenv("RAG_TEST_DOCLING_URL") // optional: enables the PDF case
	(&ingest.Worker{Store: st, Vectors: vectors, Parser: parse.New(doclingURL, 10*time.Minute, 1),
		Embedder: embedding.NewCached(sqlDB, client), Lexical: lex, Tok: tok, Defaults: cfg.Chunk, Log: log,
		Contexts: contextual.New(llm.New(llmCfg), sqlDB, tok, 6000, 4)}).Register(q)
	q.Start(ctx)
	defer q.Stop(ctx)

	retriever := retrieve.New(st, vectors, client, retrieve.Options{
		Hybrid: true, Lexical: lex, RerankCandidates: 10, Log: log,
		Rewriter: rewrite.New(llm.New(llmCfg), "multi", log),
		Reranker: rerank.New(config.RerankConfig{BaseURL: rr.URL, Format: "cohere", Timeout: 5 * time.Second}),
	})
	authn := auth.New(sqlDB, cfg.APIKeys, cfg.TenantID)
	if err := authn.EnsureDefaultTenant(ctx); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(api.New(cfg, authn, st, retriever, vectors, q, client.Model(), log).Handler())
	defer srv.Close()
	e := &env{t: t, url: srv.URL}

	// eventually polls until the vector count for a document reaches want
	// (vector cleanup runs as a background job).
	eventually := func(field, value string, want uint64) {
		t.Helper()
		var n uint64
		for i := 0; i < 100; i++ {
			if n, err = vectors.Count(ctx, field, value); err == nil && n == want {
				return
			}
			time.Sleep(50 * time.Millisecond)
		}
		t.Errorf("%s=%s: %d vectors, want %d (err %v)", field, value, n, want, err)
	}
	childCount := func(docID string) uint64 {
		var n uint64
		if err := sqlDB.QueryRowContext(ctx, "SELECT COUNT(*) FROM chunks WHERE document_id = ? AND kind = 'child'", docID).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}

	// --- dataset + documents ---
	status, ds := e.json("POST", "/v1/datasets", map[string]any{"name": "公司制度"})
	if status != http.StatusCreated {
		t.Fatalf("create dataset: %d %v", status, ds)
	}
	dsID := ds["id"].(string)

	status, up := e.upload(dsID, "员工手册.md", handbook, map[string]any{"dept": "hr", "year": 2024, "owner": "人力资源部"})
	if status != http.StatusAccepted {
		t.Fatalf("upload: %d %v", status, up)
	}
	hbID := up["document"].(map[string]any)["id"].(string)
	status, up = e.upload(dsID, "it-guide.html", itGuide, map[string]any{"dept": "it", "year": 2025, "owner": "信息技术部"})
	if status != http.StatusAccepted {
		t.Fatalf("upload: %d %v", status, up)
	}
	itID := up["document"].(map[string]any)["id"].(string)

	hb := e.waitIndexed(hbID)
	e.waitIndexed(itID)
	if hb["chunk_count"].(float64) < 2 {
		t.Errorf("handbook chunk_count = %v", hb["chunk_count"])
	}

	t.Run("duplicate upload is detected", func(t *testing.T) {
		status, out := e.upload(dsID, "copy.md", handbook, nil)
		if status != http.StatusOK || out["duplicate"] != true {
			t.Errorf("got %d %v", status, out)
		}
	})

	// PDF end to end: upload → docling → chunk → embed → Qdrant/MySQL →
	// Dify /retrieval. Needs a docling-serve (RAG_TEST_DOCLING_URL).
	t.Run("pdf goes through docling and is retrievable", func(t *testing.T) {
		if doclingURL == "" {
			t.Skip("RAG_TEST_DOCLING_URL not set")
		}
		pdf, err := os.ReadFile("../parse/testdata/sample.pdf")
		if err != nil {
			t.Fatal(err)
		}
		status, up := e.upload(dsID, "refund-policy.pdf", string(pdf), map[string]any{"dept": "sales"})
		if status != http.StatusAccepted {
			t.Fatalf("upload: %d %v", status, up)
		}
		doc := up["document"].(map[string]any)
		if doc["format"] != "pdf" || doc["status"] != "queued" {
			t.Errorf("uploaded document = %v", doc)
		}
		indexed := e.waitIndexedFor(doc["id"].(string), 5*time.Minute)
		if indexed["chunk_count"].(float64) < 2 { // at least one parent and one child
			t.Errorf("chunk_count = %v", indexed["chunk_count"])
		}

		// English question → the table row survives parsing as a Markdown table.
		recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "refund window for the annual plan",
			"retrieval_setting": map[string]any{"top_k": 3}})
		if len(recs) == 0 || recs[0]["title"] != "refund-policy.pdf" {
			t.Fatalf("top record = %v", recs)
		}
		content := recs[0]["content"].(string)
		for _, want := range []string{"Refund Policy", "| Annual", "30 days"} {
			if !strings.Contains(content, want) {
				t.Errorf("PDF content lost %q:\n%s", want, content)
			}
		}
		if md := recs[0]["metadata"].(map[string]any); md["dept"] != "sales" || md["document_id"] != doc["id"] {
			t.Errorf("metadata = %v", md)
		}

		// Chinese question against the Chinese sentence in the same PDF.
		recs = e.retrieve(map[string]any{"knowledge_id": dsID, "query": "购买后多久可以申请退款",
			"retrieval_setting":  map[string]any{"top_k": 3},
			"metadata_condition": map[string]any{"conditions": []any{map[string]any{"name": "dept", "comparison_operator": "is", "value": "sales"}}}})
		if len(recs) == 0 || !strings.Contains(recs[0]["content"].(string), "三十天内可以申请退款") {
			t.Errorf("Chinese query = %v", recs)
		}

		// Clean up so later cases see the dataset as before.
		if status, _ := e.json("DELETE", "/v1/documents/"+doc["id"].(string), nil); status != http.StatusNoContent {
			t.Errorf("delete: %d", status)
		}
		eventually(vector.FieldDocument, doc["id"].(string), 0)
	})

	t.Run("unsupported type", func(t *testing.T) {
		if status, _ := e.upload(dsID, "a.exe", "MZ", nil); status != http.StatusUnsupportedMediaType {
			t.Errorf("status %d", status)
		}
	})

	// --- Dify contract ---
	t.Run("dify save-time probe", func(t *testing.T) {
		recs := e.retrieve(map[string]any{"knowledge_id": "", "query": "",
			"retrieval_setting": map[string]any{"top_k": 1, "score_threshold": 0.0}})
		if len(recs) != 0 {
			t.Errorf("probe returned %d records", len(recs))
		}
	})

	t.Run("dify auth errors", func(t *testing.T) {
		body := `{"knowledge_id":"` + dsID + `","query":"年假"}`
		status, out := e.do("POST", "/retrieval", strings.NewReader(body), "application/json", "")
		if status != http.StatusForbidden || out["error_code"] != float64(1001) {
			t.Errorf("no header: %d %v", status, out)
		}
		status, out = e.do("POST", "/retrieval", strings.NewReader(body), "application/json", "Bearer wrong")
		if status != http.StatusForbidden || out["error_code"] != float64(1002) {
			t.Errorf("bad key: %d %v", status, out)
		}
	})

	t.Run("dify unknown knowledge", func(t *testing.T) {
		status, out := e.json("POST", "/retrieval", map[string]any{"knowledge_id": uuid.NewString(), "query": "x"})
		if status != http.StatusNotFound || out["error_code"] != float64(2001) {
			t.Errorf("got %d %v", status, out)
		}
	})

	t.Run("retrieves the right parent", func(t *testing.T) {
		recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "工龄满十年年假有几天",
			"retrieval_setting": map[string]any{"top_k": 3, "score_threshold": 0.0}})
		if len(recs) == 0 {
			t.Fatal("no records")
		}
		top := recs[0]
		if top["title"] != "员工手册.md" || !strings.Contains(top["content"].(string), "满十年不满二十年") {
			t.Errorf("top record = %v", top)
		}
		md := top["metadata"].(map[string]any)
		if md["dept"] != "hr" || md["document_id"] != hbID || md["heading"] == nil {
			t.Errorf("metadata = %v", md)
		}
		seen := map[string]bool{}
		for i, r := range recs {
			if i > 0 && r["score"].(float64) > recs[i-1]["score"].(float64) {
				t.Errorf("records not sorted by score")
			}
			id := r["metadata"].(map[string]any)["chunk_id"].(string)
			if seen[id] {
				t.Errorf("parent %s returned twice", id)
			}
			seen[id] = true
		}
	})

	t.Run("html document is searchable", func(t *testing.T) {
		recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "VPN 客户端在哪里下载",
			"retrieval_setting": map[string]any{"top_k": 1}})
		if len(recs) != 1 || recs[0]["title"] != "it-guide.html" || !strings.Contains(recs[0]["content"].(string), "内网门户") {
			t.Errorf("got %v", recs)
		}
	})

	t.Run("metadata filter is applied", func(t *testing.T) {
		for _, tc := range []struct {
			cond map[string]any
			want string
		}{
			{map[string]any{"conditions": []any{map[string]any{"name": "dept", "comparison_operator": "is", "value": "it"}}}, "it-guide.html"},
			{map[string]any{"conditions": []any{map[string]any{"name": "year", "comparison_operator": "<", "value": 2025}}}, "员工手册.md"},
			{map[string]any{"conditions": []any{map[string]any{"name": "year", "comparison_operator": "is", "value": "2025"}}}, "it-guide.html"},
			{map[string]any{"conditions": []any{map[string]any{"name": "owner", "comparison_operator": "contains", "value": "人力"}}}, "员工手册.md"},
			{map[string]any{"conditions": []any{map[string]any{"name": "owner", "comparison_operator": "start with", "value": "信息"}}}, "it-guide.html"},
			{map[string]any{"conditions": []any{map[string]any{"name": "owner", "comparison_operator": "end with", "value": "资源部"}}}, "员工手册.md"},
			{map[string]any{"conditions": []any{map[string]any{"name": "dept", "comparison_operator": "not in", "value": []any{"it"}}}}, "员工手册.md"},
			{map[string]any{"logical_operator": "and", "conditions": []any{
				map[string]any{"name": "dept", "comparison_operator": "is", "value": "hr"},
				map[string]any{"name": "year", "comparison_operator": "≥", "value": 2024}}}, "员工手册.md"},
		} {
			recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "年假", "metadata_condition": tc.cond,
				"retrieval_setting": map[string]any{"top_k": 10}})
			if len(recs) == 0 {
				t.Errorf("%v: no records", tc.cond)
			}
			for _, r := range recs {
				if r["title"] != tc.want {
					t.Errorf("%v: got record from %v", tc.cond, r["title"])
				}
			}
		}
		status, out := e.json("POST", "/retrieval", map[string]any{"knowledge_id": dsID, "query": "x",
			"metadata_condition": map[string]any{"conditions": []any{map[string]any{"name": "y", "comparison_operator": "~"}}}})
		if status != http.StatusBadRequest {
			t.Errorf("bad operator: %d %v", status, out)
		}
	})

	t.Run("score threshold", func(t *testing.T) {
		recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "完全无关的查询内容 zzz",
			"retrieval_setting": map[string]any{"top_k": 5, "score_threshold": 0.95}})
		if len(recs) != 0 {
			t.Errorf("expected nothing above 0.95, got %d", len(recs))
		}
	})

	t.Run("hybrid search finds keyword matches and reranks", func(t *testing.T) {
		status, out := e.json("POST", "/v1/search", map[string]any{"dataset_ids": []string{dsID}, "query": "差旅住宿标准", "top_k": 3})
		if status != http.StatusOK {
			t.Fatalf("search: %d %v", status, out)
		}
		results := out["results"].([]any)
		if len(results) == 0 {
			t.Fatal("no results")
		}
		top := results[0].(map[string]any)
		if !strings.Contains(top["content"].(string), "差旅住宿") {
			t.Errorf("top result = %v", top["content"])
		}
		if top["keyword_rank"] == nil {
			t.Errorf("top result has no keyword rank: %v", top)
		}
		if top["rerank_score"] == nil || top["rerank_score"] != top["score"] {
			t.Errorf("score should be the rerank score: %v", top)
		}
		for i := 1; i < len(results); i++ {
			if results[i].(map[string]any)["score"].(float64) > results[i-1].(map[string]any)["score"].(float64) {
				t.Errorf("scores not non-increasing at %d", i)
			}
		}
	})

	t.Run("reranker outage falls back to fused order", func(t *testing.T) {
		rerankBroken.Store(true)
		defer rerankBroken.Store(false)
		status, out := e.json("POST", "/v1/search", map[string]any{"dataset_ids": []string{dsID}, "query": "年假", "top_k": 3})
		if status != http.StatusOK {
			t.Fatalf("search: %d %v", status, out)
		}
		results := out["results"].([]any)
		if len(results) == 0 {
			t.Fatal("no results without reranker")
		}
		top := results[0].(map[string]any)
		if top["rerank_score"] != nil {
			t.Errorf("rerank score present although the reranker is down: %v", top)
		}
		if s := top["score"].(float64); s <= 0 || s > 1 {
			t.Errorf("fused score %v outside (0, 1]", s)
		}
	})

	t.Run("dataset reindex", func(t *testing.T) {
		status, out := e.json("POST", "/v1/datasets/"+dsID+"/reindex", nil)
		if status != http.StatusAccepted || out["queued"].(float64) < 1 {
			t.Fatalf("reindex dataset: %d %v", status, out)
		}
		e.waitIndexed(hbID)
		e.waitIndexed(itID)
	})

	t.Run("reindex uses the embedding cache", func(t *testing.T) {
		before := embedded.Load()
		status, out := e.json("POST", "/v1/documents/"+hbID+"/reindex", nil)
		if status != http.StatusAccepted {
			t.Fatalf("reindex: %d %v", status, out)
		}
		doc := e.waitIndexed(hbID)
		if doc["chunk_count"] != hb["chunk_count"] {
			t.Errorf("chunk_count changed: %v -> %v", hb["chunk_count"], doc["chunk_count"])
		}
		if n := embedded.Load() - before; n != 0 {
			t.Errorf("reindex sent %d inputs to the model, want 0 (all cached)", n)
		}
		// The old vectors are cleaned up; exactly the new children remain.
		eventually(vector.FieldDocument, hbID, childCount(hbID))
	})

	childRefs := func(docID string) map[string]string { // id -> content hash
		rows, err := sqlDB.QueryContext(ctx, "SELECT id, HEX(content_hash) FROM chunks WHERE document_id = ? AND kind = 'child'", docID)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		out := map[string]string{}
		for rows.Next() {
			var id, h string
			rows.Scan(&id, &h)
			out[id] = h
		}
		return out
	}

	t.Run("incremental update keeps unchanged chunks", func(t *testing.T) {
		before := childRefs(hbID)
		embeddedBefore := embedded.Load()
		edited := strings.Replace(handbook, "财务复核后五个工作日内打款", "财务复核后三个工作日内打款", 1)
		var buf bytes.Buffer
		mw := multipart.NewWriter(&buf)
		fw, _ := mw.CreateFormFile("file", "员工手册.md")
		fw.Write([]byte(edited))
		mw.Close()
		status, out := e.do("PUT", "/v1/documents/"+hbID+"/file", &buf, mw.FormDataContentType(), "Bearer "+apiKey)
		if status != http.StatusAccepted {
			t.Fatalf("replace file: %d %v", status, out)
		}
		e.waitIndexed(hbID)
		after := childRefs(hbID)
		kept := 0
		for id := range after {
			if _, ok := before[id]; ok {
				kept++
			}
		}
		changed := len(after) - kept
		if kept == 0 || changed == 0 {
			t.Fatalf("kept %d, changed %d of %d children; want some of each", kept, changed, len(after))
		}
		if n := embedded.Load() - embeddedBefore; n != int64(changed) {
			t.Errorf("embedded %d inputs, want %d (only the changed children)", n, changed)
		}
		eventually(vector.FieldDocument, hbID, uint64(len(after)))
		recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "报销多久打款", "retrieval_setting": map[string]any{"top_k": 3}})
		found := false
		for _, r := range recs {
			found = found || strings.Contains(r["content"].(string), "三个工作日内打款")
		}
		if !found {
			t.Errorf("edited text not retrievable: %v", recs)
		}
	})

	t.Run("metadata update re-labels vectors without re-embedding", func(t *testing.T) {
		embeddedBefore := embedded.Load()
		status, out := e.json("PATCH", "/v1/documents/"+hbID, map[string]any{"metadata": map[string]any{"dept": "finance", "year": 2024}})
		if status != http.StatusAccepted {
			t.Fatalf("patch: %d %v", status, out)
		}
		e.waitIndexed(hbID)
		if n := embedded.Load() - embeddedBefore; n != 0 {
			t.Errorf("metadata change embedded %d inputs", n)
		}
		byDept := func(dept string) int {
			return len(e.retrieve(map[string]any{"knowledge_id": dsID, "query": "年假", "retrieval_setting": map[string]any{"top_k": 10},
				"metadata_condition": map[string]any{"conditions": []any{map[string]any{"name": "dept", "comparison_operator": "is", "value": dept}}}}))
		}
		if byDept("finance") == 0 || byDept("hr") != 0 {
			t.Errorf("finance=%d hr=%d after re-labelling", byDept("finance"), byDept("hr"))
		}
	})

	t.Run("contextual retrieval", func(t *testing.T) {
		status, ds2 := e.json("POST", "/v1/datasets", map[string]any{"name": "带上下文", "chunk_config": map[string]any{"contextual": true}})
		if status != http.StatusCreated {
			t.Fatalf("create: %d %v", status, ds2)
		}
		// Sizes left out are pinned to the defaults in force (300/80 here).
		if cc, _ := ds2["chunk_config"].(map[string]any); cc["parent_tokens"] != float64(300) || cc["child_tokens"] != float64(80) {
			t.Errorf("chunk_config = %v, want parent_tokens 300 and child_tokens 80 pinned", ds2["chunk_config"])
		}
		status, up := e.upload(ds2["id"].(string), "员工手册.md", handbook, nil)
		if status != http.StatusAccepted {
			t.Fatalf("upload: %d %v", status, up)
		}
		docID := up["document"].(map[string]any)["id"].(string)
		e.waitIndexed(docID)

		var children, withContext int
		sqlDB.QueryRowContext(ctx, `SELECT COUNT(*), COUNT(context) FROM chunks WHERE document_id = ? AND kind = 'child'`, docID).Scan(&children, &withContext)
		if children == 0 || withContext != children {
			t.Errorf("%d of %d children have context", withContext, children)
		}
		var sample string
		sqlDB.QueryRowContext(ctx, `SELECT context FROM chunks WHERE document_id = ? AND kind = 'child' LIMIT 1`, docID).Scan(&sample)
		if strings.Contains(sample, "think") || !strings.HasPrefix(sample, "本段出自") {
			t.Errorf("context = %q (reasoning block must be stripped)", sample)
		}
		if got := fl.contexts.Load(); got != int64(children) {
			t.Errorf("LLM wrote %d contexts for %d children", got, children)
		}

		calls := fl.contexts.Load()
		if status, _ := e.json("POST", "/v1/documents/"+docID+"/reindex", nil); status != http.StatusAccepted {
			t.Fatalf("reindex: %d", status)
		}
		e.waitIndexed(docID)
		if n := fl.contexts.Load() - calls; n != 0 {
			t.Errorf("re-index asked the LLM for %d contexts, want 0 (cached)", n)
		}

		if status, out := e.json("POST", "/v1/datasets", map[string]any{"name": "x", "chunk_config": map[string]any{"contextual": true, "child_tokens": 10}}); status != http.StatusBadRequest {
			t.Errorf("invalid chunk_config accepted: %d %v", status, out)
		}
	})

	t.Run("query rewrite uses history and survives LLM outage", func(t *testing.T) {
		before := fl.standalone.Load()
		status, out := e.json("POST", "/v1/search", map[string]any{"dataset_ids": []string{dsID}, "query": "那满十年呢", "top_k": 3,
			"history": []any{map[string]any{"role": "user", "content": "年假有几天"}, map[string]any{"role": "assistant", "content": "满一年不满十年的为五天"}}})
		if status != http.StatusOK || len(out["results"].([]any)) == 0 {
			t.Fatalf("search with history: %d %v", status, out)
		}
		if fl.standalone.Load() == before || fl.multi.Load() == 0 {
			t.Errorf("rewriter not used: standalone=%d multi=%d", fl.standalone.Load()-before, fl.multi.Load())
		}
		top := out["results"].([]any)[0].(map[string]any)["content"].(string)
		if !strings.Contains(top, "年假") {
			t.Errorf("top result after rewrite = %q", top)
		}

		fl.broken.Store(true)
		defer fl.broken.Store(false)
		status, out = e.json("POST", "/v1/search", map[string]any{"dataset_ids": []string{dsID}, "query": "VPN 怎么连", "top_k": 3})
		if status != http.StatusOK || len(out["results"].([]any)) == 0 {
			t.Errorf("search during LLM outage: %d %v", status, out)
		}
	})

	t.Run("tenants are isolated, limited and role-checked", func(t *testing.T) {
		status, ten := e.json("POST", "/v1/admin/tenants", map[string]any{"name": "team-b", "max_datasets": 1, "max_documents": 1})
		if status != http.StatusCreated {
			t.Fatalf("create tenant: %d %v", status, ten)
		}
		tenantB := ten["id"].(string)

		// Root acts on tenant B through X-Tenant-ID and issues its admin key.
		req, _ := http.NewRequest("POST", e.url+"/v1/keys", strings.NewReader(`{"name":"b-admin","role":"admin"}`))
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("X-Tenant-ID", tenantB)
		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp.StatusCode != http.StatusCreated {
			t.Fatalf("create key for tenant B: %v %v", err, resp.Status)
		}
		var created map[string]any
		json.NewDecoder(resp.Body).Decode(&created)
		resp.Body.Close()
		adminB := created["key"].(string)

		if status, who := e.jsonAs(adminB, "GET", "/v1/whoami", nil); status != 200 || who["role"] != "admin" ||
			who["tenant"].(map[string]any)["id"] != tenantB {
			t.Fatalf("whoami: %d %v", status, who)
		}
		// Isolation: tenant B sees none of the default tenant's data.
		if status, _ := e.jsonAs(adminB, "GET", "/v1/datasets/"+dsID, nil); status != http.StatusNotFound {
			t.Errorf("cross-tenant dataset read: %d", status)
		}
		if status, out := e.jsonAs(adminB, "POST", "/retrieval", map[string]any{"knowledge_id": dsID, "query": "年假"}); status != http.StatusNotFound || out["error_code"] != float64(2001) {
			t.Errorf("cross-tenant retrieval: %d %v", status, out)
		}
		if status, out := e.jsonAs(adminB, "POST", "/v1/search", map[string]any{"dataset_ids": []string{dsID}, "query": "年假"}); status != http.StatusNotFound {
			t.Errorf("cross-tenant search: %d %v", status, out)
		}

		// Quotas.
		status, dsB := e.jsonAs(adminB, "POST", "/v1/datasets", map[string]any{"name": "b"})
		if status != http.StatusCreated {
			t.Fatalf("tenant B dataset: %d %v", status, dsB)
		}
		if status, out := e.jsonAs(adminB, "POST", "/v1/datasets", map[string]any{"name": "b2"}); status != http.StatusForbidden {
			t.Errorf("dataset quota not enforced: %d %v", status, out)
		}
		if status, _ := e.uploadAs(adminB, dsB["id"].(string), "a.md", "# A\n\n第一篇", nil); status != http.StatusAccepted {
			t.Errorf("first upload: %d", status)
		}
		if status, out := e.uploadAs(adminB, dsB["id"].(string), "b.md", "# B\n\n第二篇", nil); status != http.StatusForbidden {
			t.Errorf("document quota not enforced: %d %v", status, out)
		}

		// Roles.
		_, rk := e.jsonAs(adminB, "POST", "/v1/keys", map[string]any{"name": "dify", "role": "reader"})
		readerB := rk["key"].(string)
		if status, _ := e.jsonAs(readerB, "POST", "/v1/datasets", map[string]any{"name": "x"}); status != http.StatusForbidden {
			t.Errorf("reader created a dataset: %d", status)
		}
		if status, _ := e.jsonAs(readerB, "POST", "/v1/keys", map[string]any{"name": "x", "role": "admin"}); status != http.StatusForbidden {
			t.Errorf("reader created a key: %d", status)
		}
		if status, _ := e.jsonAs(adminB, "GET", "/v1/admin/tenants", nil); status != http.StatusForbidden {
			t.Errorf("tenant admin listed tenants: %d", status)
		}
		if status, _ := e.jsonAs(readerB, "POST", "/retrieval", map[string]any{"knowledge_id": dsB["id"], "query": "第一篇"}); status != http.StatusOK {
			t.Errorf("reader retrieval: %d", status)
		}

		// Revocation takes effect immediately.
		_, keys := e.jsonAs(adminB, "GET", "/v1/keys", nil)
		for _, k := range keys["data"].([]any) {
			if k.(map[string]any)["name"] == "dify" {
				if status, _ := e.jsonAs(adminB, "DELETE", "/v1/keys/"+k.(map[string]any)["id"].(string), nil); status != http.StatusNoContent {
					t.Errorf("revoke: %d", status)
				}
			}
		}
		if status, out := e.jsonAs(readerB, "POST", "/retrieval", map[string]any{"knowledge_id": dsB["id"], "query": "x"}); status != http.StatusForbidden || out["error_code"] != float64(1002) {
			t.Errorf("revoked key still works: %d %v", status, out)
		}

		// Rate limit: lowered to 3 rps (burst 6) on the fly by root.
		if status, out := e.json("PUT", "/v1/admin/tenants/"+tenantB, map[string]any{"name": "team-b",
			"max_datasets": 1, "max_documents": 1, "rate_limit_rps": 3}); status != http.StatusOK {
			t.Fatalf("update tenant: %d %v", status, out)
		}
		limited := 0
		for i := 0; i < 15; i++ {
			if status, _ := e.jsonAs(adminB, "GET", "/v1/usage", nil); status == http.StatusTooManyRequests {
				limited++
			}
		}
		if limited == 0 {
			t.Errorf("no request was rate limited")
		}
	})

	t.Run("console is served", func(t *testing.T) {
		resp, err := http.Get(e.url + "/ui/")
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 || !strings.Contains(string(body), "RAG 控制台") || resp.Header.Get("Content-Security-Policy") == "" {
			t.Errorf("ui: %d csp=%q", resp.StatusCode, resp.Header.Get("Content-Security-Policy"))
		}
	})

	t.Run("deleted document disappears", func(t *testing.T) {
		status, _ := e.json("DELETE", "/v1/documents/"+itID, nil)
		if status != http.StatusNoContent {
			t.Fatalf("delete: %d", status)
		}
		recs := e.retrieve(map[string]any{"knowledge_id": dsID, "query": "VPN", "retrieval_setting": map[string]any{"top_k": 10}})
		for _, r := range recs {
			if r["title"] == "it-guide.html" {
				t.Errorf("deleted document still returned")
			}
		}
		if n := childCount(itID); n != 0 {
			t.Errorf("%d chunks left behind", n)
		}
		eventually(vector.FieldDocument, itID, 0)
	})

	t.Run("deleting the dataset removes its vectors", func(t *testing.T) {
		if status, _ := e.json("DELETE", "/v1/datasets/"+dsID, nil); status != http.StatusNoContent {
			t.Fatalf("delete dataset: %d", status)
		}
		eventually(vector.FieldDataset, dsID, 0)
		status, out := e.json("POST", "/retrieval", map[string]any{"knowledge_id": dsID, "query": "年假"})
		if status != http.StatusNotFound || out["error_code"] != float64(2001) {
			t.Errorf("deleted dataset: %d %v", status, out)
		}
	})
}
