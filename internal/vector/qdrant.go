// Package vector stores child-chunk vectors in Qdrant.
//
// One point per child chunk, with the chunk's id as point id. The payload
// carries what filtering needs (tenant, dataset, document, the document's
// metadata) and nothing else; MySQL stays the source of truth, and a hit
// whose chunk row is gone is simply ignored by the retriever.
package vector

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"

	"rag/internal/config"
)

const (
	DenseVector  = "dense"
	SparseVector = "sparse" // lexical weights for hybrid search; IDF is applied by Qdrant

	FieldTenant   = "tenant_id"
	FieldDataset  = "dataset_id"
	FieldDocument = "document_id"
	FieldMetadata = "metadata"
)

type Store struct {
	c          *qdrant.Client
	collection string
	dim        int
}

type Sparse struct {
	Indices []uint32
	Values  []float32
}

type Point struct {
	ID         uuid.UUID
	Dense      []float32
	Sparse     *Sparse // optional
	TenantID   uuid.UUID
	DatasetID  uuid.UUID
	DocumentID uuid.UUID
	Metadata   map[string]any
}

type Hit struct {
	ID    uuid.UUID
	Score float32
}

func Open(ctx context.Context, cfg config.QdrantConfig, dim int) (*Store, error) {
	c, err := qdrant.NewClient(&qdrant.Config{
		Host: cfg.Host, Port: cfg.Port, APIKey: cfg.APIKey, UseTLS: cfg.UseTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant client: %w", err)
	}
	s := &Store{c: c, collection: cfg.Collection, dim: dim}
	if err := s.ensureCollection(ctx); err != nil {
		c.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error { return s.c.Close() }

func (s *Store) Ping(ctx context.Context) error {
	_, err := s.c.HealthCheck(ctx)
	return err
}

func (s *Store) ensureCollection(ctx context.Context) error {
	exists, err := s.c.CollectionExists(ctx, s.collection)
	if err != nil {
		return fmt.Errorf("qdrant: %w", err)
	}
	if !exists {
		err := s.c.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: s.collection,
			VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
				DenseVector: {
					Size:     uint64(s.dim),
					Distance: qdrant.Distance_Cosine,
					Datatype: qdrant.Datatype_Float16.Enum(), // half the memory, negligible recall loss
				},
			}),
			SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
				SparseVector: {Modifier: qdrant.Modifier_Idf.Enum()},
			}),
		})
		if err != nil && !strings.Contains(err.Error(), "already exists") {
			return fmt.Errorf("create collection %s: %w", s.collection, err)
		}
	}
	// Idempotent; tenant_id is marked as the tenant key so Qdrant can lay
	// out and search each tenant's points together.
	for _, f := range []struct {
		name   string
		tenant bool
	}{{FieldTenant, true}, {FieldDataset, false}, {FieldDocument, false}} {
		_, err := s.c.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName:   s.collection,
			Wait:             qdrant.PtrOf(true),
			FieldName:        f.name,
			FieldType:        qdrant.FieldType_FieldTypeKeyword.Enum(),
			FieldIndexParams: qdrant.NewPayloadIndexParamsKeyword(&qdrant.KeywordIndexParams{IsTenant: qdrant.PtrOf(f.tenant)}),
		})
		if err != nil {
			return fmt.Errorf("create payload index %s: %w", f.name, err)
		}
	}
	return nil
}

func (s *Store) Upsert(ctx context.Context, points []Point) error {
	const batch = 256
	for start := 0; start < len(points); start += batch {
		end := min(start+batch, len(points))
		out := make([]*qdrant.PointStruct, 0, end-start)
		for _, p := range points[start:end] {
			if len(p.Dense) != s.dim {
				return fmt.Errorf("point %s: vector has %d dims, want %d", p.ID, len(p.Dense), s.dim)
			}
			md := p.Metadata
			if md == nil {
				md = map[string]any{}
			}
			payload, err := qdrant.TryValueMap(map[string]any{
				FieldTenant:   p.TenantID.String(),
				FieldDataset:  p.DatasetID.String(),
				FieldDocument: p.DocumentID.String(),
				FieldMetadata: md,
			})
			if err != nil {
				return fmt.Errorf("point %s payload: %w", p.ID, err)
			}
			vectors := map[string]*qdrant.Vector{DenseVector: qdrant.NewVectorDense(p.Dense)}
			if p.Sparse != nil && len(p.Sparse.Indices) > 0 {
				vectors[SparseVector] = qdrant.NewVectorSparse(p.Sparse.Indices, p.Sparse.Values)
			}
			out = append(out, &qdrant.PointStruct{
				Id: qdrant.NewIDUUID(p.ID.String()), Vectors: qdrant.NewVectorsMap(vectors), Payload: payload,
			})
		}
		if _, err := s.c.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: s.collection, Wait: qdrant.PtrOf(true), Points: out,
		}); err != nil {
			return fmt.Errorf("qdrant upsert: %w", err)
		}
	}
	return nil
}

func (s *Store) DeleteIDs(ctx context.Context, ids []uuid.UUID) error {
	const batch = 1000
	for start := 0; start < len(ids); start += batch {
		end := min(start+batch, len(ids))
		pids := make([]*qdrant.PointId, 0, end-start)
		for _, id := range ids[start:end] {
			pids = append(pids, qdrant.NewIDUUID(id.String()))
		}
		if _, err := s.c.Delete(ctx, &qdrant.DeletePoints{
			CollectionName: s.collection, Wait: qdrant.PtrOf(true), Points: qdrant.NewPointsSelectorIDs(pids),
		}); err != nil {
			return fmt.Errorf("qdrant delete: %w", err)
		}
	}
	return nil
}

// DeleteWhere removes every point whose keyword field equals value
// (FieldDocument or FieldDataset).
func (s *Store) DeleteWhere(ctx context.Context, field, value string) error {
	_, err := s.c.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collection,
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelectorFilter(&qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatchKeyword(field, value)}}),
	})
	if err != nil {
		return fmt.Errorf("qdrant delete by %s: %w", field, err)
	}
	return nil
}

type SearchRequest struct {
	TenantID   uuid.UUID
	DatasetIDs []uuid.UUID
	Dense      []float32 // set exactly one of Dense and Sparse
	Sparse     *Sparse
	Filter     *qdrant.Filter // extra conditions (metadata), may be nil
	Limit      int
}

// Search returns the best-matching children, best first: by cosine
// similarity for a dense query, by BM25 for a sparse one.
func (s *Store) Search(ctx context.Context, req SearchRequest) ([]Hit, error) {
	ds := make([]string, len(req.DatasetIDs))
	for i, id := range req.DatasetIDs {
		ds[i] = id.String()
	}
	filter := &qdrant.Filter{Must: []*qdrant.Condition{
		qdrant.NewMatchKeyword(FieldTenant, req.TenantID.String()),
		qdrant.NewMatchKeywords(FieldDataset, ds...),
	}}
	if req.Filter != nil {
		filter.Must = append(filter.Must, qdrant.NewFilterAsCondition(req.Filter))
	}
	query, using := qdrant.NewQueryDense(req.Dense), DenseVector
	if req.Sparse != nil {
		if len(req.Sparse.Indices) == 0 {
			return nil, nil // no searchable terms
		}
		query, using = qdrant.NewQuerySparse(req.Sparse.Indices, req.Sparse.Values), SparseVector
	}
	points, err := s.c.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.collection,
		Query:          query,
		Using:          qdrant.PtrOf(using),
		Filter:         filter,
		Limit:          qdrant.PtrOf(uint64(req.Limit)),
		Params:         &qdrant.SearchParams{HnswEf: qdrant.PtrOf(uint64(max(req.Limit, 128)))},
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant query: %w", err)
	}
	return toHits(points)
}

func toHits(points []*qdrant.ScoredPoint) ([]Hit, error) {
	hits := make([]Hit, 0, len(points))
	for _, p := range points {
		id, err := uuid.Parse(p.GetId().GetUuid())
		if err != nil {
			return nil, fmt.Errorf("qdrant returned non-uuid point id %v", p.GetId())
		}
		hits = append(hits, Hit{ID: id, Score: p.GetScore()})
	}
	return hits, nil
}

// Count returns the exact number of points whose keyword field equals value.
func (s *Store) Count(ctx context.Context, field, value string) (uint64, error) {
	return s.c.Count(ctx, &qdrant.CountPoints{
		CollectionName: s.collection,
		Filter:         &qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatchKeyword(field, value)}},
		Exact:          qdrant.PtrOf(true),
	})
}

// DropCollection deletes the whole collection (tests only).
func (s *Store) DropCollection(ctx context.Context) error {
	return s.c.DeleteCollection(ctx, s.collection)
}

// SetDocumentMetadata replaces the metadata payload of every point of a
// document, for chunks whose vectors are kept across a re-index.
func (s *Store) SetDocumentMetadata(ctx context.Context, documentID uuid.UUID, metadata map[string]any) error {
	if metadata == nil {
		metadata = map[string]any{}
	}
	payload, err := qdrant.TryValueMap(map[string]any{FieldMetadata: metadata})
	if err != nil {
		return err
	}
	_, err = s.c.SetPayload(ctx, &qdrant.SetPayloadPoints{
		CollectionName: s.collection,
		Wait:           qdrant.PtrOf(true),
		Payload:        payload,
		PointsSelector: qdrant.NewPointsSelectorFilter(&qdrant.Filter{Must: []*qdrant.Condition{
			qdrant.NewMatchKeyword(FieldDocument, documentID.String())}}),
	})
	if err != nil {
		return fmt.Errorf("qdrant set payload: %w", err)
	}
	return nil
}
