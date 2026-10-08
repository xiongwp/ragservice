package embedding

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"rag/internal/config"
)

const testDim = 4

func testClient(url string) *Client {
	return NewClient(config.EmbeddingConfig{
		BaseURL: url, Model: "m", Dim: testDim, BatchSize: 2, Concurrency: 2, RPS: 0, Timeout: 5 * time.Second,
	})
}

// reply answers each batch in reverse order, with vector[0] = len(input text).
func reply(w http.ResponseWriter, r *http.Request, mutate func(i int, v []float32)) {
	var req embedRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	type item struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	var data []item
	for i := len(req.Input) - 1; i >= 0; i-- {
		v := []float32{float32(len(req.Input[i])), 1, 0, 0}
		if mutate != nil {
			mutate(i, v)
		}
		data = append(data, item{i, v})
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
}

func TestEmbedOrdersByIndexAcrossBatches(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reply(w, r, nil) }))
	defer srv.Close()
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	vecs, err := testClient(srv.URL).Embed(t.Context(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if int(v[0]) != len(texts[i]) {
			t.Errorf("vector %d belongs to input of length %d, want %d", i, int(v[0]), len(texts[i]))
		}
	}
}

func TestEmbedRetriesThrottling(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		reply(w, r, nil)
	}))
	defer srv.Close()
	if _, err := testClient(srv.URL).Embed(t.Context(), []string{"x"}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Errorf("calls = %d, want 2", calls.Load())
	}
}

func TestEmbedRejectsBadVectors(t *testing.T) {
	cases := map[string]http.HandlerFunc{
		"nan literal": func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,NaN,0.2,0.3]}]}`))
		},
		"zeros": func(w http.ResponseWriter, r *http.Request) { reply(w, r, func(i int, v []float32) { clear(v) }) },
		"wrong dim": func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[1,2]}]}`))
		},
		"dup index": func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[1,0,0,0]},{"index":0,"embedding":[1,0,0,0]}]}`))
		},
	}
	for name, handler := range cases {
		srv := httptest.NewServer(handler)
		_, err := testClient(srv.URL).Embed(t.Context(), []string{"x"})
		srv.Close()
		if !errors.Is(err, ErrPermanent) {
			t.Errorf("%s: want ErrPermanent, got %v", name, err)
		}
	}
}

func TestEmbedDoesNotRetryAuthErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "bad key", http.StatusUnauthorized)
	}))
	defer srv.Close()
	_, err := testClient(srv.URL).Embed(t.Context(), []string{"x"})
	if !errors.Is(err, ErrPermanent) || calls.Load() != 1 {
		t.Errorf("err = %v, calls = %d", err, calls.Load())
	}
}
