package api

import (
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"rag/internal/chunk"
	"rag/internal/db"
	"rag/internal/ingest"
	"rag/internal/parse"
	"rag/internal/queue"
	"rag/internal/store"
)

func pathID(w http.ResponseWriter, r *http.Request) (uuid.UUID, bool) {
	id, err := uuid.Parse(chi.URLParam(r, "id"))
	if err != nil {
		writeError(w, http.StatusNotFound, "not found")
		return uuid.Nil, false
	}
	return id, true
}

func (s *Server) notFoundOr(w http.ResponseWriter, r *http.Request, err error) {
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "not found")
		return
	}
	s.internalError(w, r, err, false)
}

// ---- datasets ----

func (s *Server) createDataset(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name        string            `json:"name"`
		Description string            `json:"description"`
		ChunkConfig store.ChunkConfig `json:"chunk_config"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	body.Name = strings.TrimSpace(body.Name)
	if body.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}
	cc := body.ChunkConfig
	// Sizes left out are pinned to today's defaults, so later changes to
	// CHUNK_*_TOKENS do not re-chunk (and re-embed) this dataset.
	sizes, err := chunk.Resolve(chunk.Options{ParentTokens: cc.ParentTokens, ChildTokens: cc.ChildTokens},
		chunk.Options{ParentTokens: s.cfg.Chunk.ParentTokens, ChildTokens: s.cfg.Chunk.ChildTokens})
	if err != nil {
		writeError(w, http.StatusBadRequest, "chunk_config."+err.Error())
		return
	}
	cc.ParentTokens, cc.ChildTokens = sizes.ParentTokens, sizes.ChildTokens
	if cc.Contextual && s.cfg.LLM.BaseURL == "" {
		writeError(w, http.StatusBadRequest, "chunk_config.contextual needs an LLM: set LLM_BASE_URL")
		return
	}

	d := &store.Dataset{
		ID: uuid.Must(uuid.NewV7()), TenantID: tenant(r), Name: body.Name,
		Description: body.Description, EmbeddingModel: s.model, ChunkConfig: cc,
	}
	err = db.Tx(r.Context(), s.store.DB(), func(tx *sql.Tx) error {
		if err := s.store.CheckQuota(r.Context(), tx, d.TenantID, limits(r), 1, 0, 0); err != nil {
			return err
		}
		return s.store.CreateDataset(r.Context(), tx, d)
	})
	if s.quotaError(w, err) {
		return
	}
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusCreated, d)
}

func (s *Server) listDatasets(w http.ResponseWriter, r *http.Request) {
	list, err := s.store.ListDatasets(r.Context(), tenant(r))
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": list})
}

func (s *Server) getDataset(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	d, err := s.store.GetDataset(r.Context(), tenant(r), id)
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, d)
}

func (s *Server) deleteDataset(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	err := db.Tx(r.Context(), s.store.DB(), func(tx *sql.Tx) error {
		if err := s.store.DeleteDataset(r.Context(), tx, tenant(r), id); err != nil {
			return err
		}
		return ingest.EnqueueDelete(r.Context(), tx, ingest.DeleteArgs{DatasetID: &id})
	})
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	s.queue.Notify()
	w.WriteHeader(http.StatusNoContent)
}

// ---- documents ----

// uploadDocument stores the file and enqueues indexing in one transaction:
// there is never a document without a job, or a job without its document.
func (s *Server) uploadDocument(w http.ResponseWriter, r *http.Request) {
	datasetID, ok := pathID(w, r)
	if !ok {
		return
	}
	ctx := r.Context()
	if _, err := s.store.GetDataset(ctx, tenant(r), datasetID); err != nil {
		s.notFoundOr(w, r, err)
		return
	}

	up, ok := s.readUpload(w, r)
	if !ok {
		return
	}
	name, format, data := up.name, up.format, up.data
	metadata := map[string]any{}
	// PostFormValue, not FormValue: for multipart requests r.Form lists URL
	// query values before body values, so FormValue would let ?metadata=…
	// in the URL override the form field.
	if raw := r.PostFormValue("metadata"); raw != "" {
		if err := json.Unmarshal([]byte(raw), &metadata); err != nil || metadata == nil {
			writeError(w, http.StatusBadRequest, "metadata must be a JSON object")
			return
		}
	}
	sum := up.hash

	if existing, err := s.store.FindDocumentByHash(ctx, datasetID, sum); err == nil {
		writeJSON(w, http.StatusOK, map[string]any{"document": existing, "duplicate": true})
		return
	} else if !errors.Is(err, store.ErrNotFound) {
		s.internalError(w, r, err, false)
		return
	}

	doc := &store.Document{
		ID: uuid.Must(uuid.NewV7()), TenantID: tenant(r), DatasetID: datasetID, Name: name,
		Format: format, SizeBytes: int64(len(data)), ContentHash: sum, Metadata: metadata,
	}
	err := db.Tx(ctx, s.store.DB(), func(tx *sql.Tx) error {
		if err := s.store.CheckQuota(ctx, tx, doc.TenantID, limits(r), 0, 1, int64(len(data))); err != nil {
			return err
		}
		if err := s.store.InsertDocument(ctx, tx, doc, data); err != nil {
			return err
		}
		return ingest.EnqueueIngest(ctx, tx, doc.ID)
	})
	if s.quotaError(w, err) {
		return
	}
	if db.IsDuplicate(err, "documents_dataset_hash") { // same file uploaded concurrently
		if existing, ferr := s.store.FindDocumentByHash(ctx, datasetID, sum); ferr == nil {
			writeJSON(w, http.StatusOK, map[string]any{"document": existing, "duplicate": true})
			return
		}
	}
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	s.queue.Notify()
	writeJSON(w, http.StatusAccepted, map[string]any{"document": doc, "duplicate": false})
}

func (s *Server) listDocuments(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	if _, err := s.store.GetDataset(r.Context(), tenant(r), id); err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	list, err := s.store.ListDocuments(r.Context(), tenant(r), id)
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": list})
}

func (s *Server) getDocument(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	d, err := s.store.GetDocument(r.Context(), tenant(r), id)
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, d)
}

func (s *Server) deleteDocument(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	err := db.Tx(r.Context(), s.store.DB(), func(tx *sql.Tx) error {
		if err := s.store.DeleteDocument(r.Context(), tx, tenant(r), id); err != nil {
			return err
		}
		return ingest.EnqueueDelete(r.Context(), tx, ingest.DeleteArgs{DocumentID: &id})
	})
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	s.queue.Notify()
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) reindexDocument(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	ctx := r.Context()
	if _, err := s.store.GetDocument(ctx, tenant(r), id); err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	err := db.Tx(ctx, s.store.DB(), func(tx *sql.Tx) error {
		if err := ingest.EnqueueIngest(ctx, tx, id); err != nil {
			return err
		}
		return s.store.MarkQueued(ctx, tx, id)
	})
	if errors.Is(err, queue.ErrDuplicate) {
		writeError(w, http.StatusConflict, "document is already queued or being indexed")
		return
	}
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	s.queue.Notify()
	d, err := s.store.GetDocument(ctx, tenant(r), id)
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, d)
}

// reindexDataset re-queues every document of the dataset (e.g. after an
// upgrade that changes chunking or adds keyword vectors). Documents already
// queued or running are skipped.
func (s *Server) reindexDataset(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	ctx := r.Context()
	docs, err := s.store.ListDocuments(ctx, tenant(r), id)
	if err == nil && len(docs) == 0 {
		_, err = s.store.GetDataset(ctx, tenant(r), id)
	}
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	queued, skipped := 0, 0
	for _, d := range docs {
		err := db.Tx(ctx, s.store.DB(), func(tx *sql.Tx) error {
			if err := ingest.EnqueueIngest(ctx, tx, d.ID); err != nil {
				return err
			}
			return s.store.MarkQueued(ctx, tx, d.ID)
		})
		switch {
		case errors.Is(err, queue.ErrDuplicate):
			skipped++
		case err != nil:
			s.internalError(w, r, err, false)
			return
		default:
			queued++
		}
	}
	s.queue.Notify()
	writeJSON(w, http.StatusAccepted, map[string]int{"queued": queued, "skipped": skipped})
}

type upload struct {
	name, format string
	data, hash   []byte
}

// readUpload reads and validates the multipart "file" field. On failure it
// has already written the response.
func (s *Server) readUpload(w http.ResponseWriter, r *http.Request) (*upload, bool) {
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxUploadBytes+1<<20)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "expected multipart/form-data with a \"file\" field: "+err.Error())
		return nil, false
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing \"file\" field")
		return nil, false
	}
	defer file.Close()
	if header.Size > s.cfg.MaxUploadBytes {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("file exceeds %d MB", s.cfg.MaxUploadBytes>>20))
		return nil, false
	}
	name := filepath.Base(header.Filename)
	format := parse.FormatOf(name)
	if format == "" {
		writeError(w, http.StatusUnsupportedMediaType, "unsupported file type "+filepath.Ext(name)+
			"; supported: .txt .md .html .pdf .docx .pptx .xlsx")
		return nil, false
	}
	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, http.StatusBadRequest, "read upload: "+err.Error())
		return nil, false
	}
	if len(data) == 0 {
		writeError(w, http.StatusBadRequest, "file is empty")
		return nil, false
	}
	sum := sha256.Sum256(data)
	return &upload{name: name, format: format, data: data, hash: sum[:]}, true
}

// requeue runs change (which edits the document inside tx) and re-queues
// indexing in the same transaction, then answers with the document.
func (s *Server) requeue(w http.ResponseWriter, r *http.Request, id uuid.UUID, change func(*sql.Tx) error) {
	ctx := r.Context()
	err := db.Tx(ctx, s.store.DB(), func(tx *sql.Tx) error {
		if err := change(tx); err != nil {
			return err
		}
		if err := ingest.EnqueueIngest(ctx, tx, id); err != nil {
			return err
		}
		return s.store.MarkQueued(ctx, tx, id)
	})
	if s.quotaError(w, err) {
		return
	}
	switch {
	case errors.Is(err, queue.ErrDuplicate):
		writeError(w, http.StatusConflict, "document is already queued or being indexed; retry when it finishes")
		return
	case errors.Is(err, store.ErrNotFound):
		writeError(w, http.StatusNotFound, "not found")
		return
	case db.IsDuplicate(err, "documents_dataset_hash"):
		writeError(w, http.StatusConflict, "another document in this dataset has identical content")
		return
	case err != nil:
		s.internalError(w, r, err, false)
		return
	}
	s.queue.Notify()
	d, err := s.store.GetDocument(ctx, tenant(r), id)
	if err != nil {
		s.notFoundOr(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, d)
}

// updateDocumentMetadata replaces the metadata and re-indexes; only the
// vector payloads change, nothing is re-embedded.
func (s *Server) updateDocumentMetadata(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	var body struct {
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&body); err != nil || body.Metadata == nil {
		writeError(w, http.StatusBadRequest, `body must be {"metadata": {...}}`)
		return
	}
	s.requeue(w, r, id, func(tx *sql.Tx) error {
		return s.store.UpdateDocumentMetadata(r.Context(), tx, tenant(r), id, body.Metadata)
	})
}

// replaceDocumentFile uploads a new version of a document. Chunks whose
// text did not change keep their vectors.
func (s *Server) replaceDocumentFile(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	up, ok := s.readUpload(w, r)
	if !ok {
		return
	}
	defer r.MultipartForm.RemoveAll() //nolint:errcheck
	s.requeue(w, r, id, func(tx *sql.Tx) error {
		old, err := s.store.GetDocument(r.Context(), tenant(r), id)
		if err != nil {
			return err
		}
		if err := s.store.CheckQuota(r.Context(), tx, tenant(r), limits(r), 0, 0, int64(len(up.data))-old.SizeBytes); err != nil {
			return err
		}
		return s.store.ReplaceDocumentFile(r.Context(), tx, tenant(r), id, up.name, up.format, up.hash, up.data)
	})
}
