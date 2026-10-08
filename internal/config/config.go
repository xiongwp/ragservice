// Package config loads ragd settings from environment variables.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// EmbeddingDim is the size of the Qdrant collection's dense vector. Changing
// it means a new collection; the service refuses to start with a different
// EMBEDDING_DIM so a mismatch fails loudly instead of at upsert time.
const EmbeddingDim = 1024

type Config struct {
	HTTPAddr       string
	MySQLDSN       string
	Qdrant         QdrantConfig
	APIKeys        []string
	TenantID       uuid.UUID
	MaxUploadBytes int64

	Workers int

	Embedding EmbeddingConfig

	DoclingURL         string
	DoclingTimeout     time.Duration
	DoclingConcurrency int
	// TextFallbackEncoding is assumed for legacy (non-UTF-8) text files whose
	// encoding cannot be detected with confidence.
	TextFallbackEncoding string

	Chunk ChunkDefaults

	// SearchMode is "hybrid" (vector + BM25 fused with RRF) or "dense".
	SearchMode string
	Rerank     RerankConfig

	LLM LLMConfig
	// QueryRewrite: "off", "multi" (paraphrases searched alongside the
	// query) or "hyde" (a hypothetical answer searched by vector). Needs LLM.
	QueryRewrite string
	// ContextTokens caps how much of the document is shown to the LLM when
	// writing chunk context (Contextual Retrieval).
	ContextTokens int
}

// LLMConfig: an empty BaseURL disables every LLM feature.
type LLMConfig struct {
	BaseURL     string
	APIKey      string
	Model       string
	Timeout     time.Duration
	Concurrency int
	RPS         float64
}

// RerankConfig: an empty BaseURL disables reranking.
type RerankConfig struct {
	BaseURL    string
	APIKey     string
	Model      string
	Format     string // "cohere" (Cohere/Jina/SiliconFlow/vLLM) or "tei"
	Timeout    time.Duration
	Candidates int // parents sent to the reranker per query
}

type QdrantConfig struct {
	Host       string
	Port       int // gRPC port
	APIKey     string
	UseTLS     bool
	Collection string
}

type EmbeddingConfig struct {
	BaseURL     string
	APIKey      string
	Model       string
	Dim         int
	BatchSize   int
	Concurrency int
	RPS         float64
	Timeout     time.Duration
}

type ChunkDefaults struct {
	ParentTokens int
	ChildTokens  int
}

func Load() (*Config, error) {
	var errs []string
	str := func(key, def string) string {
		if v, ok := os.LookupEnv(key); ok && v != "" {
			return v
		}
		return def
	}
	num := func(key string, def int) int {
		v := str(key, "")
		if v == "" {
			return def
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", key, err))
		}
		return n
	}
	float := func(key string, def float64) float64 {
		v := str(key, "")
		if v == "" {
			return def
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", key, err))
		}
		return f
	}
	dur := func(key string, def time.Duration) time.Duration {
		v := str(key, "")
		if v == "" {
			return def
		}
		d, err := time.ParseDuration(v)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", key, err))
		}
		return d
	}

	c := &Config{
		HTTPAddr: str("RAG_HTTP_ADDR", ":8080"),
		MySQLDSN: str("RAG_MYSQL_DSN", ""),
		Qdrant: QdrantConfig{
			Host:       str("QDRANT_HOST", "localhost"),
			Port:       num("QDRANT_GRPC_PORT", 6334),
			APIKey:     str("QDRANT_API_KEY", ""),
			UseTLS:     str("QDRANT_TLS", "") == "true",
			Collection: str("QDRANT_COLLECTION", "rag_chunks"),
		},
		MaxUploadBytes: int64(num("RAG_MAX_UPLOAD_MB", 50)) << 20,
		Workers:        num("RAG_WORKERS", 4),
		Embedding: EmbeddingConfig{
			BaseURL:     strings.TrimRight(str("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"), "/"),
			APIKey:      str("EMBEDDING_API_KEY", ""),
			Model:       str("EMBEDDING_MODEL", "BAAI/bge-m3"),
			Dim:         num("EMBEDDING_DIM", EmbeddingDim),
			BatchSize:   num("EMBEDDING_BATCH_SIZE", 32),
			Concurrency: num("EMBEDDING_CONCURRENCY", 4),
			RPS:         float("EMBEDDING_RPS", 10),
			Timeout:     dur("EMBEDDING_TIMEOUT", 60*time.Second),
		},
		DoclingURL:           strings.TrimRight(str("DOCLING_URL", ""), "/"),
		DoclingTimeout:       dur("DOCLING_TIMEOUT", time.Hour),
		DoclingConcurrency:   num("DOCLING_CONCURRENCY", 1),
		TextFallbackEncoding: str("RAG_TEXT_FALLBACK_ENCODING", "gb18030"),
		Chunk: ChunkDefaults{
			ParentTokens: num("CHUNK_PARENT_TOKENS", 800),
			ChildTokens:  num("CHUNK_CHILD_TOKENS", 200),
		},
		SearchMode: str("RAG_SEARCH_MODE", "hybrid"),
		Rerank: RerankConfig{
			BaseURL:    strings.TrimRight(str("RERANK_BASE_URL", ""), "/"),
			APIKey:     str("RERANK_API_KEY", ""),
			Model:      str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
			Format:     str("RERANK_FORMAT", "cohere"),
			Timeout:    dur("RERANK_TIMEOUT", 15*time.Second),
			Candidates: num("RERANK_CANDIDATES", 30),
		},
		LLM: LLMConfig{
			BaseURL:     strings.TrimRight(str("LLM_BASE_URL", ""), "/"),
			APIKey:      str("LLM_API_KEY", ""),
			Model:       str("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
			Timeout:     dur("LLM_TIMEOUT", 60*time.Second),
			Concurrency: num("LLM_CONCURRENCY", 4),
			RPS:         float("LLM_RPS", 5),
		},
		QueryRewrite:  str("RAG_QUERY_REWRITE", "off"),
		ContextTokens: num("CONTEXT_DOC_TOKENS", 6000),
	}

	for _, k := range strings.Split(str("RAG_API_KEYS", ""), ",") {
		if k = strings.TrimSpace(k); k != "" {
			c.APIKeys = append(c.APIKeys, k)
		}
	}
	tenant, err := uuid.Parse(str("RAG_TENANT_ID", "00000000-0000-0000-0000-000000000001"))
	if err != nil {
		errs = append(errs, fmt.Sprintf("RAG_TENANT_ID: %v", err))
	}
	c.TenantID = tenant

	if c.MySQLDSN == "" {
		errs = append(errs, "RAG_MYSQL_DSN is required, e.g. rag:rag@tcp(localhost:3306)/rag")
	}
	if len(c.APIKeys) == 0 {
		errs = append(errs, "RAG_API_KEYS is required (comma-separated root keys; per-tenant keys are created through the API)")
	}
	if c.Embedding.Dim != EmbeddingDim {
		errs = append(errs, fmt.Sprintf("EMBEDDING_DIM=%d but the vector collection is %d-dim; pick a %d-dim model", c.Embedding.Dim, EmbeddingDim, EmbeddingDim))
	}
	if c.Embedding.BatchSize < 1 || c.Embedding.Concurrency < 1 || c.Workers < 1 {
		errs = append(errs, "EMBEDDING_BATCH_SIZE, EMBEDDING_CONCURRENCY and RAG_WORKERS must be >= 1")
	}
	if c.Chunk.ChildTokens < 32 || c.Chunk.ParentTokens < c.Chunk.ChildTokens {
		errs = append(errs, "need CHUNK_CHILD_TOKENS >= 32 and CHUNK_PARENT_TOKENS >= CHUNK_CHILD_TOKENS")
	}
	if c.SearchMode != "hybrid" && c.SearchMode != "dense" {
		errs = append(errs, "RAG_SEARCH_MODE must be hybrid or dense")
	}
	switch c.QueryRewrite {
	case "off":
	case "multi", "hyde":
		if c.LLM.BaseURL == "" {
			errs = append(errs, "RAG_QUERY_REWRITE="+c.QueryRewrite+" needs LLM_BASE_URL")
		}
	default:
		errs = append(errs, "RAG_QUERY_REWRITE must be off, multi or hyde")
	}
	if c.Rerank.Format != "cohere" && c.Rerank.Format != "tei" {
		errs = append(errs, "RERANK_FORMAT must be cohere or tei")
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("invalid config:\n  %s", strings.Join(errs, "\n  "))
	}
	return c, nil
}
