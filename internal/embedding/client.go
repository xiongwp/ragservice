// Package embedding calls an OpenAI-compatible /embeddings endpoint
// (OpenAI, SiliconFlow, DashScope compatible mode, TEI, vLLM, ...).
package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/time/rate"

	"rag/internal/config"
)

// ErrPermanent wraps failures a retry cannot fix (bad key, bad model, bad input).
var ErrPermanent = errors.New("permanent embedding error")

type Client struct {
	cfg     config.EmbeddingConfig
	http    *http.Client
	limiter *rate.Limiter
	sem     chan struct{} // caps in-flight requests across all callers
}

func NewClient(cfg config.EmbeddingConfig) *Client {
	limit := rate.Inf
	if cfg.RPS > 0 {
		limit = rate.Limit(cfg.RPS)
	}
	return &Client{
		cfg:     cfg,
		http:    &http.Client{Timeout: cfg.Timeout},
		limiter: rate.NewLimiter(limit, max(1, cfg.Concurrency)),
		sem:     make(chan struct{}, cfg.Concurrency),
	}
}

func (c *Client) Model() string { return c.cfg.Model }

// Embed returns one vector per input, in input order. Inputs are sent in
// batches of BatchSize with at most Concurrency requests in flight.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	g, ctx := errgroup.WithContext(ctx)
	for start := 0; start < len(texts); start += c.cfg.BatchSize {
		end := min(start+c.cfg.BatchSize, len(texts))
		g.Go(func() error {
			vecs, err := c.embedBatch(ctx, texts[start:end])
			if err != nil {
				return fmt.Errorf("inputs %d..%d: %w", start, end-1, err)
			}
			copy(out[start:end], vecs)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

type embedRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
}

type embedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	select {
	case c.sem <- struct{}{}:
		defer func() { <-c.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	body, err := json.Marshal(embedRequest{Model: c.cfg.Model, Input: texts, EncodingFormat: "float"})
	if err != nil {
		return nil, err
	}

	const maxAttempts = 6
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if err := c.limiter.Wait(ctx); err != nil {
			return nil, err
		}
		vecs, retryAfter, err := c.do(ctx, body, len(texts))
		if err == nil {
			return vecs, nil
		}
		if errors.Is(err, ErrPermanent) || ctx.Err() != nil {
			return nil, err
		}
		lastErr = err
		wait := retryAfter
		if wait == 0 {
			backoff := 500 * time.Millisecond << attempt
			wait = min(backoff, 20*time.Second) + rand.N(250*time.Millisecond)
		}
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return nil, fmt.Errorf("gave up after %d attempts: %w", maxAttempts, lastErr)
}

// do performs one request. A non-zero duration is the server's Retry-After.
func (c *Client) do(ctx context.Context, body []byte, n int) ([][]float32, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 256<<20))
	if err != nil {
		return nil, 0, err
	}
	if resp.StatusCode != http.StatusOK {
		msg := fmt.Errorf("HTTP %d: %.300s", resp.StatusCode, raw)
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			secs, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
			return nil, time.Duration(secs) * time.Second, msg
		}
		if (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) && c.cfg.APIKey == "" {
			return nil, 0, fmt.Errorf("%w: EMBEDDING_API_KEY is not set (%v)", ErrPermanent, msg)
		}
		return nil, 0, fmt.Errorf("%w: %v", ErrPermanent, msg)
	}

	var out embedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		// Python servers can emit bare NaN/Infinity, which is not JSON. Retrying
		// returns the same garbage, so fail now and say why.
		if nonFiniteJSON.Match(raw) {
			return nil, 0, fmt.Errorf("%w: model returned non-finite values (NaN/Infinity)", ErrPermanent)
		}
		return nil, 0, fmt.Errorf("decode response: %w", err)
	}
	// Place vectors by their declared index, never by arrival order, and make
	// sure every input got exactly one well-formed vector.
	vecs := make([][]float32, n)
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= n || vecs[d.Index] != nil {
			return nil, 0, fmt.Errorf("%w: response has bad or duplicate index %d", ErrPermanent, d.Index)
		}
		if err := Validate(d.Embedding, c.cfg.Dim); err != nil {
			return nil, 0, fmt.Errorf("%w: input %d: %v", ErrPermanent, d.Index, err)
		}
		vecs[d.Index] = d.Embedding
	}
	for i, v := range vecs {
		if v == nil {
			return nil, 0, fmt.Errorf("response is missing input %d of %d", i, n)
		}
	}
	return vecs, 0, nil
}

var nonFiniteJSON = regexp.MustCompile(`[\[,:]\s*-?(NaN|Infinity)\b`)

// Validate rejects vectors that would silently corrupt the index.
func Validate(v []float32, dim int) error {
	if len(v) != dim {
		return fmt.Errorf("vector has %d dims, want %d", len(v), dim)
	}
	var norm float64
	for i, x := range v {
		f := float64(x)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("vector has non-finite value at %d", i)
		}
		norm += f * f
	}
	if norm == 0 {
		return errors.New("vector is all zeros")
	}
	return nil
}
