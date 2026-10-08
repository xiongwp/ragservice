// Package api serves the Dify external-knowledge endpoint and the management API.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/google/uuid"
	"golang.org/x/time/rate"

	"rag/internal/auth"
	"rag/internal/config"
	"rag/internal/queue"
	"rag/internal/retrieve"
	"rag/internal/store"
	"rag/internal/vector"
)

type Server struct {
	cfg       *config.Config
	authn     *auth.Authenticator
	limits    *tenantLimiter
	store     *store.Store
	retriever *retrieve.Retriever
	vectors   *vector.Store
	queue     *queue.Queue
	model     string // embedding model new datasets are bound to
	log       *slog.Logger
}

func New(cfg *config.Config, authn *auth.Authenticator, st *store.Store, rt *retrieve.Retriever, vectors *vector.Store,
	q *queue.Queue, model string, log *slog.Logger) *Server {
	return &Server{cfg: cfg, authn: authn, limits: &tenantLimiter{limiters: map[uuid.UUID]*rate.Limiter{}},
		store: st, retriever: rt, vectors: vectors, queue: q, model: model, log: log}
}

func (s *Server) Handler() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID, s.logRequests, middleware.Recoverer)

	r.Get("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := s.store.DB().PingContext(r.Context()); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "mysql unavailable"})
			return
		}
		if err := s.vectors.Ping(r.Context()); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "qdrant unavailable"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	r.Get("/", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/ui/", http.StatusFound) })
	r.Handle("/ui/*", uiHandler())

	// Dify appends /retrieval to the endpoint configured in its console.
	r.With(s.auth(true)).Post("/retrieval", s.difyRetrieval)

	r.Route("/v1", func(r chi.Router) {
		r.Use(s.auth(false))

		// reader
		r.Get("/whoami", s.whoami)
		r.Get("/usage", s.usage)
		r.Get("/datasets", s.listDatasets)
		r.Get("/datasets/{id}", s.getDataset)
		r.Get("/datasets/{id}/documents", s.listDocuments)
		r.Get("/documents/{id}", s.getDocument)
		r.Post("/search", s.search)

		r.Group(func(r chi.Router) {
			r.Use(require(auth.Editor))
			r.Post("/datasets", s.createDataset)
			r.Delete("/datasets/{id}", s.deleteDataset)
			r.Post("/datasets/{id}/reindex", s.reindexDataset)
			r.Post("/datasets/{id}/documents", s.uploadDocument)
			r.Delete("/documents/{id}", s.deleteDocument)
			r.Patch("/documents/{id}", s.updateDocumentMetadata)
			r.Put("/documents/{id}/file", s.replaceDocumentFile)
			r.Post("/documents/{id}/reindex", s.reindexDocument)
		})
		r.Group(func(r chi.Router) {
			r.Use(require(auth.Admin))
			r.Get("/keys", s.listKeys)
			r.Post("/keys", s.createKey)
			r.Delete("/keys/{id}", s.revokeKey)
		})
		r.Route("/admin", func(r chi.Router) {
			r.Use(require(auth.Root))
			r.Get("/tenants", s.listTenants)
			r.Post("/tenants", s.saveTenant)
			r.Put("/tenants/{id}", s.saveTenant)
		})
	})
	return r
}

type principalKey struct{}

// principal is the authenticated caller; auth() guarantees it is set.
func principal(r *http.Request) *auth.Principal {
	return r.Context().Value(principalKey{}).(*auth.Principal)
}

// tenant is the tenant every query of this request is scoped to.
func tenant(r *http.Request) uuid.UUID { return principal(r).Tenant.ID }

// auth authenticates the caller and applies the tenant's rate limit. dify
// selects Dify's error format (codes 1001/1002) for /retrieval.
func (s *Server) auth(dify bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			token, ok := auth.BearerToken(r.Header.Get("Authorization"))
			if !ok {
				if dify {
					writeDifyError(w, http.StatusForbidden, 1001, "Invalid Authorization header format. Expected 'Bearer <api-key>' format.")
				} else {
					writeError(w, http.StatusUnauthorized, "missing bearer token")
				}
				return
			}
			p, err := s.authn.Authenticate(r.Context(), token, r.Header.Get("X-Tenant-ID"))
			if errors.Is(err, auth.ErrUnauthorized) {
				if dify {
					writeDifyError(w, http.StatusForbidden, 1002, "Authorization failed")
				} else {
					writeError(w, http.StatusUnauthorized, err.Error())
				}
				return
			}
			if err != nil {
				s.internalError(w, r, err, dify)
				return
			}
			if !s.limits.allow(p.Tenant.ID, p.Tenant.RateLimitRPS) {
				w.Header().Set("Retry-After", "1")
				if dify {
					writeDifyError(w, http.StatusTooManyRequests, 4290, "rate limit exceeded")
				} else {
					writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
				}
				return
			}
			next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), principalKey{}, p)))
		})
	}
}

func require(role auth.Role) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if principal(r).Role < role {
				writeError(w, http.StatusForbidden, "this needs the "+role.String()+" role")
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// tenantLimiter keeps one token bucket per tenant (burst: two seconds' worth).
type tenantLimiter struct {
	mu       sync.Mutex
	limiters map[uuid.UUID]*rate.Limiter
}

func (l *tenantLimiter) allow(tenant uuid.UUID, rps float64) bool {
	if rps <= 0 {
		return true
	}
	l.mu.Lock()
	lim, ok := l.limiters[tenant]
	if !ok {
		lim = rate.NewLimiter(rate.Limit(rps), max(1, int(2*rps)))
		l.limiters[tenant] = lim
	} else if lim.Limit() != rate.Limit(rps) { // limit changed by an admin
		lim.SetLimit(rate.Limit(rps))
		lim.SetBurst(max(1, int(2*rps)))
	}
	l.mu.Unlock()
	return lim.Allow()
}

func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
		start := time.Now()
		next.ServeHTTP(ww, r)
		if r.URL.Path == "/healthz" {
			return
		}
		s.log.Info("http", "method", r.Method, "path", r.URL.Path, "status", ww.Status(),
			"took", time.Since(start).Round(time.Microsecond), "req_id", middleware.GetReqID(r.Context()))
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeDifyError(w http.ResponseWriter, status, code int, msg string) {
	writeJSON(w, status, map[string]any{"error_code": code, "error_msg": msg})
}
