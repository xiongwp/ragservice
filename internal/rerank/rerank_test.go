package rerank

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"rag/internal/config"
)

func serve(t *testing.T, body string, check func(map[string]any)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if check != nil {
			check(req)
		}
		_, _ = w.Write([]byte(body))
	}))
}

func TestCohereFormat(t *testing.T) {
	srv := serve(t, `{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]}`, func(req map[string]any) {
		if req["model"] != "m" || req["query"] != "q" || len(req["documents"].([]any)) != 2 {
			t.Errorf("request = %v", req)
		}
	})
	defer srv.Close()
	c := New(config.RerankConfig{BaseURL: srv.URL, Model: "m", Format: "cohere", Timeout: time.Second})
	scores, err := c.Rerank(t.Context(), "q", []string{"a", "b"})
	if err != nil || scores[0] != 0.2 || scores[1] != 0.9 {
		t.Errorf("scores = %v, %v", scores, err)
	}
}

func TestTEIFormat(t *testing.T) {
	srv := serve(t, `[{"index":0,"score":0.7},{"index":1,"score":0.1}]`, func(req map[string]any) {
		if len(req["texts"].([]any)) != 2 {
			t.Errorf("request = %v", req)
		}
	})
	defer srv.Close()
	c := New(config.RerankConfig{BaseURL: srv.URL, Format: "tei", Timeout: time.Second})
	scores, err := c.Rerank(t.Context(), "q", []string{"a", "b"})
	if err != nil || scores[0] != 0.7 || scores[1] != 0.1 {
		t.Errorf("scores = %v, %v", scores, err)
	}
}

func TestMissingScoreIsAnError(t *testing.T) {
	srv := serve(t, `{"results":[{"index":0,"relevance_score":0.9}]}`, nil)
	defer srv.Close()
	c := New(config.RerankConfig{BaseURL: srv.URL, Format: "cohere", Timeout: time.Second})
	if _, err := c.Rerank(t.Context(), "q", []string{"a", "b"}); err == nil {
		t.Error("want error when a document gets no score")
	}
}

func TestDisabled(t *testing.T) {
	if New(config.RerankConfig{}) != nil {
		t.Error("empty base URL should disable reranking")
	}
}
