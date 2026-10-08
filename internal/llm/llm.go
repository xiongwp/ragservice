// Package llm calls an OpenAI-compatible /chat/completions endpoint for
// the small generation tasks retrieval needs (chunk context, query
// rewriting). It is deliberately minimal: one prompt in, text out.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"

	"golang.org/x/time/rate"

	"rag/internal/config"
)

var ErrPermanent = errors.New("permanent LLM error")

type Client struct {
	cfg     config.LLMConfig
	http    *http.Client
	limiter *rate.Limiter
	sem     chan struct{}
}

// New returns nil when no LLM is configured.
func New(cfg config.LLMConfig) *Client {
	if cfg.BaseURL == "" {
		return nil
	}
	limit := rate.Inf
	if cfg.RPS > 0 {
		limit = rate.Limit(cfg.RPS)
	}
	return &Client{
		cfg:     cfg,
		http:    &http.Client{Timeout: cfg.Timeout},
		limiter: rate.NewLimiter(limit, max(1, cfg.Concurrency)),
		sem:     make(chan struct{}, max(1, cfg.Concurrency)),
	}
}

func (c *Client) Model() string { return c.cfg.Model }

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Complete sends one user prompt and returns the reply text. maxTokens caps
// the reply; attempts bounds retries (online callers pass 1).
func (c *Client) Complete(ctx context.Context, prompt string, maxTokens, attempts int) (string, error) {
	select {
	case c.sem <- struct{}{}:
		defer func() { <-c.sem }()
	case <-ctx.Done():
		return "", ctx.Err()
	}
	body, err := json.Marshal(map[string]any{
		"model":       c.cfg.Model,
		"messages":    []message{{Role: "user", Content: prompt}},
		"max_tokens":  maxTokens,
		"temperature": 0,
		"stream":      false,
	})
	if err != nil {
		return "", err
	}
	var lastErr error
	for attempt := 0; attempt < max(1, attempts); attempt++ {
		if attempt > 0 {
			wait := min(time.Second<<attempt, 20*time.Second) + rand.N(250*time.Millisecond)
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
		if err := c.limiter.Wait(ctx); err != nil {
			return "", err
		}
		text, retryAfter, err := c.do(ctx, body)
		if err == nil {
			return text, nil
		}
		if errors.Is(err, ErrPermanent) || ctx.Err() != nil {
			return "", err
		}
		lastErr = err
		if retryAfter > 0 {
			select {
			case <-time.After(retryAfter):
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
	}
	return "", lastErr
}

func (c *Client) do(ctx context.Context, body []byte) (string, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", 0, err
	}
	if resp.StatusCode != http.StatusOK {
		msg := fmt.Errorf("HTTP %d: %.300s", resp.StatusCode, raw)
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			secs, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
			return "", time.Duration(secs) * time.Second, msg
		}
		return "", 0, fmt.Errorf("%w: %v", ErrPermanent, msg)
	}
	var out struct {
		Choices []struct {
			Message message `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", 0, fmt.Errorf("decode response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", 0, errors.New("response has no choices")
	}
	return strings.TrimSpace(stripThinking(out.Choices[0].Message.Content)), 0, nil
}

// stripThinking drops a leading <think>…</think> block that reasoning
// models (Qwen3, DeepSeek-R1) put before the answer.
func stripThinking(s string) string {
	t := strings.TrimSpace(s)
	if strings.HasPrefix(t, "<think>") {
		if end := strings.Index(t, "</think>"); end >= 0 {
			return t[end+len("</think>"):]
		}
	}
	return s
}
