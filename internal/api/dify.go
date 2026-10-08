package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"rag/internal/retrieve"
	"rag/internal/rewrite"
	"rag/internal/store"
)

// Contract: api/services/knowledge/external/service.py in Dify.
type difyRequest struct {
	KnowledgeID      string `json:"knowledge_id"`
	Query            string `json:"query"`
	RetrievalSetting struct {
		TopK           *int     `json:"top_k"`
		ScoreThreshold *float64 `json:"score_threshold"`
	} `json:"retrieval_setting"`
	MetadataCondition *retrieve.MetadataCondition `json:"metadata_condition"`
}

type difyRecord struct {
	Content  string         `json:"content"`
	Score    float64        `json:"score"`
	Title    string         `json:"title"`
	Metadata map[string]any `json:"metadata"`
}

func (s *Server) difyRetrieval(w http.ResponseWriter, r *http.Request) {
	var req difyRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		writeDifyError(w, http.StatusBadRequest, 4000, "invalid JSON body: "+err.Error())
		return
	}
	// When an endpoint is saved, Dify probes it with knowledge_id "" and
	// query "" and treats anything but 403/404/502 as reachable.
	if req.KnowledgeID == "" || strings.TrimSpace(req.Query) == "" {
		writeJSON(w, http.StatusOK, map[string]any{"records": []difyRecord{}})
		return
	}
	id, err := uuid.Parse(req.KnowledgeID)
	if err != nil {
		writeDifyError(w, http.StatusNotFound, 2001, "The knowledge does not exist")
		return
	}
	ds, err := s.store.GetDataset(r.Context(), tenant(r), id)
	if errors.Is(err, store.ErrNotFound) {
		writeDifyError(w, http.StatusNotFound, 2001, "The knowledge does not exist")
		return
	}
	if err != nil {
		s.internalError(w, r, err, true)
		return
	}
	if ds.EmbeddingModel != s.model {
		writeDifyError(w, http.StatusInternalServerError, 5001,
			"dataset was indexed with "+ds.EmbeddingModel+" but the service now embeds with "+s.model)
		return
	}

	q := retrieve.Query{DatasetIDs: []uuid.UUID{id}, Text: req.Query, Filter: req.MetadataCondition}
	if req.RetrievalSetting.TopK != nil {
		q.TopK = *req.RetrievalSetting.TopK
	}
	if req.RetrievalSetting.ScoreThreshold != nil {
		q.ScoreThreshold = *req.RetrievalSetting.ScoreThreshold
	}
	results, err := s.retriever.Search(r.Context(), tenant(r), q)
	var fe *retrieve.FilterError
	if errors.As(err, &fe) {
		writeDifyError(w, http.StatusBadRequest, 4001, "metadata_condition: "+fe.Error())
		return
	}
	if err != nil {
		s.internalError(w, r, err, true)
		return
	}

	records := make([]difyRecord, len(results))
	for i, res := range results {
		md := make(map[string]any, len(res.Metadata)+4)
		for k, v := range res.Metadata {
			md[k] = v
		}
		// Dify overwrites dataset_id/score/title in this map with its own values.
		md["document_id"] = res.DocumentID.String()
		md["chunk_id"] = res.ParentID.String()
		md["matched_chunk_id"] = res.ChunkID.String()
		if res.Heading != "" {
			md["heading"] = res.Heading
		}
		records[i] = difyRecord{Content: res.Content, Score: res.Score, Title: res.DocumentName, Metadata: md}
	}
	writeJSON(w, http.StatusOK, map[string]any{"records": records})
}

func (s *Server) internalError(w http.ResponseWriter, r *http.Request, err error, dify bool) {
	s.log.Error("request failed", "path", r.URL.Path, "err", err)
	if dify {
		writeDifyError(w, http.StatusInternalServerError, 5000, "internal error")
		return
	}
	writeError(w, http.StatusInternalServerError, "internal error")
}

// search is the debugging twin of /retrieval: same pipeline, several
// datasets at once, and the per-stage ranks in the response.
func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	var req struct {
		DatasetIDs        []uuid.UUID                 `json:"dataset_ids"`
		Query             string                      `json:"query"`
		TopK              int                         `json:"top_k"`
		ScoreThreshold    float64                     `json:"score_threshold"`
		MetadataCondition *retrieve.MetadataCondition `json:"metadata_condition"`
		History           []rewrite.Turn              `json:"history"` // earlier turns, oldest first
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	for _, id := range req.DatasetIDs {
		ds, err := s.store.GetDataset(r.Context(), tenant(r), id)
		if err != nil {
			s.notFoundOr(w, r, err)
			return
		}
		if ds.EmbeddingModel != s.model {
			writeError(w, http.StatusConflict, "dataset "+id.String()+" was indexed with "+ds.EmbeddingModel)
			return
		}
	}
	start := time.Now()
	results, err := s.retriever.Search(r.Context(), tenant(r), retrieve.Query{
		DatasetIDs: req.DatasetIDs, Text: req.Query, TopK: req.TopK,
		ScoreThreshold: req.ScoreThreshold, Filter: req.MetadataCondition, History: req.History,
	})
	var fe *retrieve.FilterError
	if errors.As(err, &fe) {
		writeError(w, http.StatusBadRequest, "metadata_condition: "+fe.Error())
		return
	}
	if err != nil {
		s.internalError(w, r, err, false)
		return
	}
	type hit struct {
		DocumentID   uuid.UUID      `json:"document_id"`
		DocumentName string         `json:"document_name"`
		ChunkID      uuid.UUID      `json:"chunk_id"`
		MatchedChunk uuid.UUID      `json:"matched_chunk_id"`
		Heading      string         `json:"heading"`
		Content      string         `json:"content"`
		Score        float64        `json:"score"`
		VectorRank   int            `json:"vector_rank,omitempty"`
		KeywordRank  int            `json:"keyword_rank,omitempty"`
		RerankScore  *float64       `json:"rerank_score,omitempty"`
		Metadata     map[string]any `json:"metadata"`
	}
	hits := make([]hit, len(results))
	for i, res := range results {
		hits[i] = hit{DocumentID: res.DocumentID, DocumentName: res.DocumentName, ChunkID: res.ParentID,
			MatchedChunk: res.ChunkID, Heading: res.Heading, Content: res.Content, Score: res.Score,
			VectorRank: res.VectorRank, KeywordRank: res.KeywordRank, RerankScore: res.RerankScore,
			Metadata: res.Metadata}
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": hits, "took_ms": time.Since(start).Milliseconds()})
}
