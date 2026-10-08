// Package rerank scores (query, passage) pairs with a cross-encoder service
// such as bge-reranker-v2-m3.
//
// Two wire formats are supported:
//   - cohere: POST {base}/rerank {model, query, documents} → {results: [{index, relevance_score}]}
//     (Cohere, Jina, SiliconFlow, vLLM, Xinference)
//   - tei:    POST {base}/rerank {query, texts} → [{index, score}] (HuggingFace TEI)
package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"rag/internal/config"
)

type Client struct {
	cfg  config.RerankConfig
	http *http.Client
}

// New returns nil when reranking is not configured.
func New(cfg config.RerankConfig) *Client {
	if cfg.BaseURL == "" {
		return nil
	}
	return &Client{cfg: cfg, http: &http.Client{Timeout: cfg.Timeout}}
}

// Rerank returns one relevance score per document, aligned with docs.
func (c *Client) Rerank(ctx context.Context, query string, docs []string) ([]float64, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	var body any
	if c.cfg.Format == "tei" {
		body = map[string]any{"query": query, "texts": docs, "raw_scores": false, "truncate": true}
	} else {
		body = map[string]any{"model": c.cfg.Model, "query": query, "documents": docs,
			"top_n": len(docs), "return_documents": false}
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	var resp []byte
	for attempt := 0; ; attempt++ {
		var retry bool
		resp, retry, err = c.do(ctx, raw)
		if err == nil || !retry || attempt == 1 {
			break
		}
		select {
		case <-time.After(300 * time.Millisecond):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if err != nil {
		return nil, err
	}

	type item struct {
		Index          int      `json:"index"`
		RelevanceScore *float64 `json:"relevance_score"`
		Score          *float64 `json:"score"`
	}
	var items []item
	if c.cfg.Format == "tei" {
		err = json.Unmarshal(resp, &items)
	} else {
		var out struct {
			Results []item `json:"results"`
		}
		err = json.Unmarshal(resp, &out)
		items = out.Results
	}
	if err != nil {
		return nil, fmt.Errorf("rerank: decode response: %w", err)
	}

	scores := make([]float64, len(docs))
	seen := make([]bool, len(docs))
	for _, it := range items {
		s := it.RelevanceScore
		if s == nil {
			s = it.Score
		}
		if it.Index < 0 || it.Index >= len(docs) || seen[it.Index] || s == nil || math.IsNaN(*s) {
			return nil, fmt.Errorf("rerank: bad result %+v", it)
		}
		scores[it.Index], seen[it.Index] = *s, true
	}
	for i, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("rerank: no score for document %d of %d", i, len(docs))
		}
	}
	return scores, nil
}

func (c *Client) do(ctx context.Context, body []byte) ([]byte, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, ctx.Err() == nil, fmt.Errorf("rerank: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, true, err
	}
	if resp.StatusCode != http.StatusOK {
		retry := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		return nil, retry, fmt.Errorf("rerank: HTTP %d: %.300s", resp.StatusCode, raw)
	}
	return raw, false, nil
}
