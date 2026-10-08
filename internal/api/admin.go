package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"rag/internal/auth"
	"rag/internal/store"
)

func limits(r *http.Request) store.Limits {
	t := principal(r).Tenant
	return store.Limits{MaxDatasets: t.MaxDatasets, MaxDocuments: t.MaxDocuments, MaxStorageMB: t.MaxStorageMB}
}

// quotaError answers 403 for a quota violation and reports whether it did.
func (s *Server) quotaError(w http.ResponseWriter, err error) bool {
	var qe *store.QuotaError
	if errors.As(err, &qe) {
		writeError(w, http.StatusForbidden, qe.Msg)
		return true
	}
	return false
}

func (s *Server) whoami(w http.ResponseWriter, r *http.Request) {
	p := principal(r)
	writeJSON(w, http.StatusOK, map[string]any{
		"tenant": p.Tenant, "role": p.Role.String(), "key_name": p.KeyName,
		"features": map[string]any{
			"search_mode": s.cfg.SearchMode, "rerank": s.cfg.Rerank.BaseURL != "", "llm": s.cfg.LLM.BaseURL != "",
			"query_rewrite": s.cfg.QueryRewrite, "embedding_model": s.model, "docling": s.cfg.DoclingURL != "",
		},
	})
}

func (s *Server) usage(w http.ResponseWriter, r *http.Request) {
	u, err := s.store.Usage(r.Context(), s.store.DB(), tenant(r))
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"usage": u, "limits": principal(r).Tenant})
}

// ---- keys (tenant admins) ----

func (s *Server) listKeys(w http.ResponseWriter, r *http.Request) {
	keys, err := s.authn.ListKeys(r.Context(), tenant(r))
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": keys})
}

func (s *Server) createKey(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name string `json:"name"`
		Role string `json:"role"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	role, ok := auth.ParseRole(body.Role)
	if !ok {
		writeError(w, http.StatusBadRequest, "role must be admin, editor or reader")
		return
	}
	if role > principal(r).Role {
		writeError(w, http.StatusForbidden, "cannot create a key with more rights than your own")
		return
	}
	if strings.TrimSpace(body.Name) == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	secret, key, err := s.authn.CreateKey(r.Context(), tenant(r), strings.TrimSpace(body.Name), role)
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"key": secret, "info": key,
		"note": "store this key now; it cannot be shown again"})
}

func (s *Server) revokeKey(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	if err := s.authn.RevokeKey(r.Context(), tenant(r), id); err != nil {
		if errors.Is(err, auth.ErrNotFound) {
			writeError(w, http.StatusNotFound, "not found")
			return
		}
		s.internalError(w, r, err, false)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- tenants (root) ----

func (s *Server) listTenants(w http.ResponseWriter, r *http.Request) {
	ts, err := s.authn.ListTenants(r.Context())
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": ts})
}

// saveTenant creates (POST) or updates (PUT /tenants/{id}) a tenant.
func (s *Server) saveTenant(w http.ResponseWriter, r *http.Request) {
	var t auth.Tenant
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&t); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	t.Name = strings.TrimSpace(t.Name)
	if t.Name == "" || t.RateLimitRPS < 0 || t.MaxDatasets < 0 || t.MaxDocuments < 0 || t.MaxStorageMB < 0 {
		writeError(w, http.StatusBadRequest, "name is required and limits must be >= 0 (0 = unlimited)")
		return
	}
	status := http.StatusCreated
	if idStr := chi.URLParam(r, "id"); idStr != "" {
		id, err := uuid.Parse(idStr)
		if err != nil {
			writeError(w, http.StatusNotFound, "not found")
			return
		}
		if _, err := s.authn.GetTenant(r.Context(), id); err != nil {
			writeError(w, http.StatusNotFound, "not found")
			return
		}
		t.ID, status = id, http.StatusOK
	} else {
		t.ID = uuid.Nil
	}
	if err := s.authn.SaveTenant(r.Context(), &t); err != nil {
		s.internalError(w, r, err, false)
		return
	}
	saved, err := s.authn.GetTenant(r.Context(), t.ID)
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, status, saved)
}
