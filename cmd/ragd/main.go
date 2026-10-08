// ragd is the RAG service: HTTP API, Dify external-knowledge endpoint and
// indexing workers in one binary.
//
//	ragd            run migrations, then serve
//	ragd migrate    run migrations and exit
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

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

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var err error
	switch cmd := arg(1); cmd {
	case "", "serve":
		err = serve(ctx, log)
	case "migrate":
		dsn := os.Getenv("RAG_MYSQL_DSN")
		if dsn == "" {
			err = errors.New("RAG_MYSQL_DSN is required")
			break
		}
		err = db.Migrate(ctx, dsn, log)
	default:
		err = fmt.Errorf("unknown command %q (want serve or migrate)", cmd)
	}
	if err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func arg(i int) string {
	if len(os.Args) > i {
		return os.Args[i]
	}
	return ""
}

func serve(ctx context.Context, log *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if err := db.WaitReady(ctx, cfg.MySQLDSN, 2*time.Minute, log); err != nil {
		return err
	}
	if err := db.Migrate(ctx, cfg.MySQLDSN, log); err != nil {
		return fmt.Errorf("migrate: %w", err)
	}
	sqlDB, err := db.Open(ctx, cfg.MySQLDSN)
	if err != nil {
		return err
	}
	defer sqlDB.Close()
	vectors, err := vector.Open(ctx, cfg.Qdrant, cfg.Embedding.Dim)
	if err != nil {
		return err
	}
	defer vectors.Close()

	tok, err := chunk.NewTokenizer()
	if err != nil {
		return err
	}
	lex, err := lexical.New()
	if err != nil {
		return err
	}
	authn := auth.New(sqlDB, cfg.APIKeys, cfg.TenantID)
	if err := authn.EnsureDefaultTenant(ctx); err != nil {
		return err
	}
	st := store.New(sqlDB)
	// Indexing (chunk context) and queries (rewriting) get separate LLM
	// clients so a big upload cannot starve interactive search.
	var contexts *contextual.Generator
	var rewriter *rewrite.Rewriter
	if indexLLM := llm.New(cfg.LLM); indexLLM != nil {
		contexts = contextual.New(indexLLM, sqlDB, tok, cfg.ContextTokens, cfg.LLM.Concurrency)
		if cfg.QueryRewrite != "off" {
			rewriter = rewrite.New(llm.New(cfg.LLM), cfg.QueryRewrite, log)
		}
	}
	embedClient := embedding.NewClient(cfg.Embedding)
	// Queries get their own client so they never queue behind indexing
	// batches on the local rate limiter / concurrency cap.
	queryClient := embedding.NewClient(cfg.Embedding)

	parser := parse.New(cfg.DoclingURL, cfg.DoclingTimeout, cfg.DoclingConcurrency)
	if err := parser.SetFallbackEncoding(cfg.TextFallbackEncoding); err != nil {
		return fmt.Errorf("RAG_TEXT_FALLBACK_ENCODING: %w", err)
	}
	q := queue.New(sqlDB, cfg.Workers, log.With("component", "queue"))
	(&ingest.Worker{
		Store:    st,
		Vectors:  vectors,
		Parser:   parser,
		Embedder: embedding.NewCached(sqlDB, embedClient),
		Lexical:  lex,
		Contexts: contexts,
		Tok:      tok,
		Defaults: cfg.Chunk,
		Log:      log,
	}).Register(q)
	q.Start(ctx)

	srv := &http.Server{
		Addr: cfg.HTTPAddr,
		Handler: api.New(cfg, authn, st, retrieve.New(st, vectors, queryClient, retrieve.Options{
			Hybrid:           cfg.SearchMode == "hybrid",
			Lexical:          lex,
			Reranker:         rerank.New(cfg.Rerank),
			Rewriter:         rewriter,
			RerankCandidates: cfg.Rerank.Candidates,
			Log:              log,
		}), vectors, q, embedClient.Model(), log).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	if cfg.Embedding.APIKey == "" && !isLocal(cfg.Embedding.BaseURL) {
		log.Warn("EMBEDDING_API_KEY is empty; the embedding service will reject requests and documents will fail to index",
			"embedding_base_url", cfg.Embedding.BaseURL)
	}
	errc := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", cfg.HTTPAddr, "embedding_model", embedClient.Model(), "workers", cfg.Workers,
			"docling", cfg.DoclingURL != "",
			"search_mode", cfg.SearchMode, "rerank", cfg.Rerank.BaseURL != "",
			"llm", cfg.LLM.BaseURL != "", "query_rewrite", cfg.QueryRewrite)
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("http shutdown", "err", err)
	}
	// Running jobs get the same grace period; unfinished ones are retried later.
	if err := q.Stop(shutdownCtx); err != nil {
		log.Error("worker shutdown", "err", err)
	}
	return nil
}

// isLocal reports whether a model endpoint runs on this host or a private
// network, where an empty API key is normal (TEI, vLLM, devmodels).
func isLocal(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	host := u.Hostname()
	if host == "localhost" || !strings.Contains(host, ".") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && (ip.IsLoopback() || ip.IsPrivate())
}
