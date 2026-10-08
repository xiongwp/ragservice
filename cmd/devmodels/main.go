// devmodels is a stand-in model server for local development and demos:
// OpenAI-style /v1/embeddings and Cohere-style /v1/rerank, with no API key
// and no GPU. Embeddings hash character bigrams into 1024 dimensions, so
// texts sharing words land close together; that is enough to click through
// the console, not to judge retrieval quality.
//
//	go run ./cmd/devmodels -addr :9900
//	EMBEDDING_BASE_URL=http://localhost:9900/v1 EMBEDDING_MODEL=dev-bigram RERANK_BASE_URL=http://localhost:9900/v1 ragd
package main

import (
	"encoding/json"
	"flag"
	"hash/fnv"
	"log"
	"math"
	"net/http"
	"strings"
)

const dim = 1024

func embed(text string) []float32 {
	v := make([]float32, dim)
	r := []rune(strings.ToLower(text))
	for i := 0; i+1 < len(r); i++ {
		h := fnv.New32a()
		h.Write([]byte(string(r[i : i+2])))
		v[h.Sum32()%dim]++
	}
	var n float64
	for _, x := range v {
		n += float64(x * x)
	}
	if n == 0 {
		v[0], n = 1, 1
	}
	for i := range v {
		v[i] /= float32(math.Sqrt(n))
	}
	return v
}

func main() {
	addr := flag.String("addr", ":9900", "listen address")
	flag.Parse()

	http.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		data := make([]map[string]any, len(req.Input))
		for i, t := range req.Input {
			data[i] = map[string]any{"index": i, "embedding": embed(t)}
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	})
	http.HandleFunc("POST /v1/rerank", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Query     string   `json:"query"`
			Documents []string `json:"documents"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		q := embed(req.Query)
		results := make([]map[string]any, len(req.Documents))
		for i, d := range req.Documents {
			var dot float64
			for j, x := range embed(d) {
				dot += float64(x * q[j])
			}
			results[i] = map[string]any{"index": i, "relevance_score": dot}
		}
		json.NewEncoder(w).Encode(map[string]any{"results": results})
	})
	log.Printf("devmodels listening on %s (embedding dim %d)", *addr, dim)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
