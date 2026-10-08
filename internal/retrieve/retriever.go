// Package retrieve answers queries against indexed chunks.
//
// Pipeline:
//  0. Rewrite (optional): an LLM adds paraphrases or a hypothetical answer.
//  1. Recall: vector searches and (in hybrid mode) BM25 searches run in
//     parallel in Qdrant, one per query variant, with tenant, dataset and
//     metadata filters applied inside each search.
//  2. Fuse: Reciprocal Rank Fusion over child chunks. RRF looks only at
//     ranks, so it needs no tuning between cosine and BM25 score scales.
//  3. Assemble: load the children from MySQL with their parent and
//     document, and collapse to parents (the context the LLM reads), each
//     at its best child's position. MySQL is the source of truth: a vector
//     whose chunk row is gone is dropped here.
//  4. Rerank (optional): a cross-encoder rescores the top parents.
//
// Reported scores are always in [0, 1] and non-increasing down the list,
// so Dify's score_threshold stays meaningful: the reranker's relevance
// score; else in hybrid mode the RRF score divided by its maximum; else the
// cosine similarity.
package retrieve

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"

	"github.com/google/uuid"
	"golang.org/x/sync/errgroup"

	"rag/internal/lexical"
	"rag/internal/rerank"
	"rag/internal/rewrite"
	"rag/internal/store"
	"rag/internal/vector"
)

type QueryEmbedder interface {
	Model() string
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type Query struct {
	DatasetIDs     []uuid.UUID
	Text           string
	TopK           int
	ScoreThreshold float64
	Filter         *MetadataCondition
	History        []rewrite.Turn // earlier conversation turns, for query rewriting
}

type Result struct {
	ChunkID      uuid.UUID // the best-matching child
	ParentID     uuid.UUID // the returned context (== ChunkID for a parentless chunk)
	DocumentID   uuid.UUID
	DocumentName string
	Heading      string
	Content      string // parent content, which is what the LLM should read
	Score        float64
	Metadata     map[string]any // document metadata

	VectorRank, KeywordRank int      // 1-based rank of the child in each list; 0 = not found
	RerankScore             *float64 // set when a reranker ran
}

type Options struct {
	Hybrid           bool
	Lexical          *lexical.Analyzer // required when Hybrid
	Reranker         *rerank.Client    // nil disables reranking
	Rewriter         *rewrite.Rewriter // nil disables query rewriting
	RerankCandidates int
	Log              *slog.Logger
}

type Retriever struct {
	store    *store.Store
	vectors  *vector.Store
	embedder QueryEmbedder
	opts     Options
}

func New(st *store.Store, vectors *vector.Store, embedder QueryEmbedder, opts Options) *Retriever {
	if opts.Log == nil {
		opts.Log = slog.Default()
	}
	if opts.RerankCandidates <= 0 {
		opts.RerankCandidates = 30
	}
	return &Retriever{store: st, vectors: vectors, embedder: embedder, opts: opts}
}

const (
	maxTopK = 100
	rrfK    = 60 // the constant from Cormack et al.; damps the weight of the very top ranks
)

func (r *Retriever) Search(ctx context.Context, tenant uuid.UUID, q Query) ([]Result, error) {
	if strings.TrimSpace(q.Text) == "" || len(q.DatasetIDs) == 0 {
		return []Result{}, nil
	}
	topK := q.TopK
	if topK <= 0 {
		topK = 5
	}
	topK = min(topK, maxTopK)
	// Several children of one parent can match; over-fetch so that after
	// collapsing to parents there are still enough distinct results.
	want := topK
	if r.opts.Reranker != nil {
		want = max(topK, r.opts.RerankCandidates)
	}
	limit := min(max(want*4, 40), 500)

	filter, err := BuildFilter(ctx, q.Filter, func(ctx context.Context, field, like string) ([]string, error) {
		return r.store.DocumentsWhereMetadata(ctx, tenant, q.DatasetIDs, field, like)
	})
	if err != nil {
		return nil, err
	}

	exp := rewrite.Expansion{Query: q.Text}
	if r.opts.Rewriter != nil {
		exp = r.opts.Rewriter.Expand(ctx, q.Text, q.History)
	}
	queries := append([]string{exp.Query}, exp.Queries...)
	denseTexts := append(append([]string{}, queries...), exp.Passages...)
	var sparseTexts []string
	if r.opts.Hybrid {
		sparseTexts = queries
	}

	dense := make([][]vector.Hit, len(denseTexts))
	sparse := make([][]vector.Hit, len(sparseTexts))
	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		vecs, err := r.embedder.Embed(gctx, denseTexts)
		if err != nil {
			return fmt.Errorf("embed query: %w", err)
		}
		dg, dctx := errgroup.WithContext(gctx)
		for i, v := range vecs {
			dg.Go(func() (err error) {
				dense[i], err = r.vectors.Search(dctx, vector.SearchRequest{
					TenantID: tenant, DatasetIDs: q.DatasetIDs, Dense: v, Filter: filter, Limit: limit,
				})
				return err
			})
		}
		return dg.Wait()
	})
	for i, text := range sparseTexts {
		g.Go(func() (err error) {
			sparse[i], err = r.vectors.Search(gctx, vector.SearchRequest{
				TenantID: tenant, DatasetIDs: q.DatasetIDs, Sparse: r.opts.Lexical.Query(text), Filter: filter, Limit: limit,
			})
			return err
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	results, err := r.assemble(ctx, tenant, fuse(dense, sparse), want)
	if err != nil {
		return nil, err
	}
	if r.opts.Reranker != nil && len(results) > 0 {
		r.rerank(ctx, exp.Query, results)
	}

	out := []Result{}
	for _, res := range results {
		if res.Score < q.ScoreThreshold || len(out) == topK {
			break
		}
		out = append(out, res)
	}
	return out, nil
}

type candidate struct {
	id                      uuid.UUID
	score                   float64
	vectorRank, keywordRank int // best rank across the lists of each kind
}

// fuse merges ranked lists with RRF and returns candidates best first,
// scored in [0, 1]. A single vector list keeps its cosine similarities.
func fuse(dense, sparse [][]vector.Hit) []candidate {
	if len(dense) == 1 && len(sparse) == 0 {
		out := make([]candidate, len(dense[0]))
		for i, h := range dense[0] {
			out[i] = candidate{id: h.ID, score: math.Max(0, float64(h.Score)), vectorRank: i + 1}
		}
		return out
	}
	byID := map[uuid.UUID]*candidate{}
	var order []*candidate
	lists := 0
	add := func(hits []vector.Hit, keyword bool) {
		if len(hits) > 0 {
			lists++
		}
		for i, h := range hits {
			c, ok := byID[h.ID]
			if !ok {
				c = &candidate{id: h.ID}
				byID[h.ID] = c
				order = append(order, c)
			}
			rank := i + 1
			c.score += 1.0 / float64(rrfK+rank)
			best := &c.vectorRank
			if keyword {
				best = &c.keywordRank
			}
			if *best == 0 || rank < *best {
				*best = rank
			}
		}
	}
	for _, l := range dense {
		add(l, false)
	}
	for _, l := range sparse {
		add(l, true)
	}
	// Normalise by the best achievable score: rank 1 in every non-empty list.
	best := float64(lists) / float64(rrfK+1)
	out := make([]candidate, len(order))
	for i, c := range order {
		c.score = math.Min(1, c.score/best) // float rounding can land a hair above 1
		out[i] = *c
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].score > out[j].score })
	return out
}

// assemble joins candidates with MySQL and collapses them to at most n
// parents, best first.
func (r *Retriever) assemble(ctx context.Context, tenant uuid.UUID, cands []candidate, n int) ([]Result, error) {
	ids := make([]uuid.UUID, len(cands))
	for i, c := range cands {
		ids[i] = c.id
	}
	views, err := r.store.ChunkViews(ctx, tenant, ids)
	if err != nil {
		return nil, err
	}
	out := []Result{}
	seen := map[uuid.UUID]bool{}
	for _, c := range cands {
		v, ok := views[c.id]
		if !ok || seen[v.ParentID] {
			continue
		}
		seen[v.ParentID] = true
		out = append(out, Result{
			ChunkID: v.ChunkID, ParentID: v.ParentID, DocumentID: v.DocumentID, DocumentName: v.DocumentName,
			Heading: v.Heading, Content: v.Content, Score: c.score, Metadata: v.Metadata,
			VectorRank: c.vectorRank, KeywordRank: c.keywordRank,
		})
		if len(out) == n {
			break
		}
	}
	return out, nil
}

// rerank rescores results in place and re-sorts them. If the reranker
// fails, the fused order is kept: a degraded answer beats no answer.
func (r *Retriever) rerank(ctx context.Context, query string, results []Result) {
	docs := make([]string, len(results))
	for i, res := range results {
		docs[i] = res.Content
		if res.Heading != "" {
			docs[i] = res.Heading + "\n" + res.Content
		}
	}
	scores, err := r.opts.Reranker.Rerank(ctx, query, docs)
	if err != nil {
		r.opts.Log.Warn("rerank failed; using fused order", "err", err)
		return
	}
	for i := range results {
		results[i].RerankScore = &scores[i]
		results[i].Score = math.Max(0, math.Min(1, scores[i]))
	}
	sort.SliceStable(results, func(i, j int) bool { return *results[i].RerankScore > *results[j].RerankScore })
}
