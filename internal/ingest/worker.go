// Package ingest runs the indexing pipeline for one document as a queued
// job: parse → clean → chunk → embed → upsert vectors → swap chunk rows.
//
// Parallelism lives at two levels: the queue runs RAG_WORKERS documents at
// once, and within a document the embedder fans batches out to
// EMBEDDING_CONCURRENCY requests (shared by all documents, so total pressure
// on the model service stays bounded however many jobs run).
//
// Consistency between MySQL and Qdrant: new vectors are written first, then
// the chunk rows are swapped in one MySQL transaction, which also enqueues
// deletion of the old vectors. Until that commit the new vectors have no
// rows and the retriever ignores them; after it the old ones have no rows
// and are ignored until the cleanup job removes them.
package ingest

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"

	"rag/internal/chunk"
	"rag/internal/config"
	"rag/internal/contextual"
	"rag/internal/embedding"
	"rag/internal/lexical"
	"rag/internal/llm"
	"rag/internal/parse"
	"rag/internal/queue"
	"rag/internal/store"
	"rag/internal/vector"
)

const (
	KindIngest        = "ingest_document"
	KindDeleteVectors = "delete_vectors"
)

type IngestArgs struct {
	DocumentID uuid.UUID `json:"document_id"`
}

// DeleteArgs selects vectors to remove: explicit ids, or everything of a
// document or dataset.
type DeleteArgs struct {
	PointIDs   []uuid.UUID `json:"point_ids,omitempty"`
	DocumentID *uuid.UUID  `json:"document_id,omitempty"`
	DatasetID  *uuid.UUID  `json:"dataset_id,omitempty"`
}

// EnqueueIngest returns queue.ErrDuplicate if the document is already
// queued or being indexed.
func EnqueueIngest(ctx context.Context, tx *sql.Tx, documentID uuid.UUID) error {
	return queue.Insert(ctx, tx, KindIngest, IngestArgs{documentID},
		queue.InsertOpts{UniqueKey: "ingest:" + documentID.String(), MaxAttempts: 3})
}

func EnqueueDelete(ctx context.Context, tx *sql.Tx, args DeleteArgs) error {
	return queue.Insert(ctx, tx, KindDeleteVectors, args, queue.InsertOpts{MaxAttempts: 10})
}

type Embedder interface {
	Model() string
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type Worker struct {
	Store    *store.Store
	Vectors  *vector.Store
	Parser   *parse.Parser
	Embedder Embedder
	Lexical  *lexical.Analyzer     // BM25 sparse vectors for hybrid search
	Contexts *contextual.Generator // nil when no LLM is configured
	Tok      chunk.Tokenizer
	Defaults config.ChunkDefaults
	Log      *slog.Logger
}

func (w *Worker) Register(q *queue.Queue) {
	q.Register(KindIngest, 2*time.Hour, w.ingest) // includes queueing for docling
	q.Register(KindDeleteVectors, 5*time.Minute, w.deleteVectors)
}

// errPermanent marks failures where retrying cannot help.
var errPermanent = errors.New("permanent")

func (w *Worker) ingest(ctx context.Context, job *queue.Job) error {
	var args IngestArgs
	if err := json.Unmarshal(job.Args, &args); err != nil {
		return queue.Cancel(err)
	}
	id := args.DocumentID
	start := time.Now()
	n, err := w.run(ctx, id)
	if errors.Is(err, store.ErrNotFound) {
		w.Log.Info("document deleted before indexing finished", "document_id", id)
		return nil
	}
	if err != nil {
		permanent := errors.Is(err, errPermanent) || errors.Is(err, parse.ErrUnsupported) ||
			errors.Is(err, embedding.ErrPermanent)
		status := "retrying"
		if permanent || job.Attempt >= job.MaxAttempts {
			status = "failed"
		}
		if uerr := w.Store.SetProgress(context.WithoutCancel(ctx), id, status, "", err.Error()); uerr != nil {
			w.Log.Error("record failure", "document_id", id, "err", uerr)
		}
		w.Log.Warn("indexing failed", "document_id", id, "attempt", job.Attempt, "status", status, "err", err)
		if permanent {
			return queue.Cancel(err)
		}
		return err
	}
	w.Log.Info("indexed document", "document_id", id, "chunks", n, "took", time.Since(start).Round(time.Millisecond))
	return nil
}

func (w *Worker) run(ctx context.Context, id uuid.UUID) (int, error) {
	doc, ds, data, err := w.Store.LoadForIngest(ctx, id)
	if err != nil {
		return 0, err
	}
	if ds.EmbeddingModel != w.Embedder.Model() {
		return 0, fmt.Errorf("%w: dataset uses embedding model %q but the service is configured with %q",
			errPermanent, ds.EmbeddingModel, w.Embedder.Model())
	}
	if ds.ChunkConfig.Contextual && w.Contexts == nil {
		return 0, fmt.Errorf("%w: dataset uses contextual retrieval but no LLM is configured (LLM_BASE_URL)", errPermanent)
	}
	// Datasets created since sizes are pinned carry both; older ones fall
	// back to CHUNK_*_TOKENS, which may since have changed. Checked before
	// parsing, which can take minutes.
	opts, err := chunk.Resolve(
		chunk.Options{ParentTokens: ds.ChunkConfig.ParentTokens, ChildTokens: ds.ChunkConfig.ChildTokens},
		chunk.Options{ParentTokens: w.Defaults.ParentTokens, ChildTokens: w.Defaults.ChildTokens})
	if err != nil {
		return 0, fmt.Errorf("%w: dataset chunk_config: %v", errPermanent, err)
	}
	stage := func(s string) error { return w.Store.SetProgress(ctx, id, "processing", s, "") }

	if err := stage("parsing"); err != nil {
		return 0, err
	}
	md, err := w.Parser.Parse(ctx, doc.Name, doc.Format, data)
	if err != nil {
		return 0, err
	}
	text := chunk.Clean(md)
	if text == "" {
		return 0, fmt.Errorf("%w: no text could be extracted", errPermanent)
	}

	if err := stage("chunking"); err != nil {
		return 0, err
	}
	parents := chunk.Split(text, opts, w.Tok)
	title := strings.TrimSuffix(doc.Name, filepath.Ext(doc.Name))

	var contexts [][]string // per parent, per child
	if ds.ChunkConfig.Contextual {
		if err := stage("contextualizing"); err != nil {
			return 0, err
		}
		var items []contextual.Item
		for _, p := range parents {
			for _, c := range p.Children {
				items = append(items, contextual.Item{Parent: p.Content, Child: c.Content})
			}
		}
		flat, err := w.Contexts.Contexts(ctx, title, text, items)
		if err != nil {
			if errors.Is(err, llm.ErrPermanent) {
				err = fmt.Errorf("%w: %v", errPermanent, err)
			}
			return 0, err
		}
		for _, p := range parents {
			contexts = append(contexts, flat[:len(p.Children)])
			flat = flat[len(p.Children):]
		}
	}

	// Chunks whose content (and context) did not change keep their id, and
	// with it their vector: only the difference is embedded and written.
	old, err := w.Store.ChunkRefs(ctx, doc.ID)
	if err != nil {
		return 0, err
	}
	plan := buildRows(title, parents, contexts, old)

	if err := stage("embedding"); err != nil {
		return 0, err
	}
	var inputs []string
	for _, i := range plan.changed {
		inputs = append(inputs, plan.inputs[i])
	}
	vecs, err := w.Embedder.Embed(ctx, inputs)
	if err != nil {
		return 0, fmt.Errorf("embed: %w", err)
	}
	if len(vecs) != len(inputs) {
		return 0, fmt.Errorf("embedder returned %d vectors for %d inputs", len(vecs), len(inputs))
	}
	// vecs[j] belongs to the j-th changed child (the embedder returns vectors
	// in input order, checked by index), so the pairing cannot drift.
	points := make([]vector.Point, len(plan.changed))
	newIDs := make([]uuid.UUID, len(plan.changed))
	for j, i := range plan.changed {
		r := plan.rows[plan.childRow[i]]
		points[j] = vector.Point{
			ID: r.ID, Dense: vecs[j], Sparse: w.Lexical.Document(plan.inputs[i]),
			TenantID: doc.TenantID, DatasetID: doc.DatasetID, DocumentID: doc.ID, Metadata: doc.Metadata,
		}
		newIDs[j] = r.ID
	}

	if err := stage("indexing"); err != nil {
		return 0, err
	}
	if err := w.Vectors.Upsert(ctx, points); err != nil {
		return 0, err
	}
	if plan.reusedChildren > 0 {
		// Kept vectors still carry the metadata they were indexed with.
		if err := w.Vectors.SetDocumentMetadata(ctx, doc.ID, doc.Metadata); err != nil {
			return 0, err
		}
	}
	err = w.Store.ReplaceChunks(ctx, doc, plan.rows, plan.tokens, func(tx *sql.Tx) error {
		if len(plan.obsolete) == 0 {
			return nil
		}
		return EnqueueDelete(ctx, tx, DeleteArgs{PointIDs: plan.obsolete})
	})
	if err != nil {
		// The rows never landed, so the new vectors are unreachable; remove them.
		if derr := w.Vectors.DeleteIDs(context.WithoutCancel(ctx), newIDs); derr != nil {
			w.Log.Warn("remove vectors of failed indexing", "document_id", id, "err", derr)
		}
		return 0, err
	}
	w.Log.Info("chunk diff", "document_id", id, "children", len(plan.inputs),
		"reused", plan.reusedChildren, "embedded", len(plan.changed), "removed", len(plan.obsolete))
	return len(plan.rows), nil
}

func (w *Worker) deleteVectors(ctx context.Context, job *queue.Job) error {
	var args DeleteArgs
	if err := json.Unmarshal(job.Args, &args); err != nil {
		return queue.Cancel(err)
	}
	switch {
	case args.DatasetID != nil:
		return w.Vectors.DeleteWhere(ctx, vector.FieldDataset, args.DatasetID.String())
	case args.DocumentID != nil:
		return w.Vectors.DeleteWhere(ctx, vector.FieldDocument, args.DocumentID.String())
	default:
		return w.Vectors.DeleteIDs(ctx, args.PointIDs)
	}
}

type rowPlan struct {
	rows           []store.Chunk
	inputs         []string // embedding input of each child, in child order
	childRow       []int    // index into rows of each child
	changed        []int    // children (by child index) that need a new vector
	reusedChildren int
	obsolete       []uuid.UUID // old children whose vectors must go
	tokens         int
}

// buildRows flattens parents and children into rows (parents first, so
// positions read top to bottom). A chunk whose hash matches one of the
// document's existing chunks of the same kind takes over its id; for a
// child that means its vector is still valid. Children hash their full
// embedding input, so a changed title, heading or context counts as a
// change.
func buildRows(title string, parents []chunk.Parent, contexts [][]string, old []store.ChunkRef) rowPlan {
	pool := map[string][]uuid.UUID{} // kind + hash -> unclaimed old ids
	for _, r := range old {
		k := r.Kind + string(r.ContentHash)
		pool[k] = append(pool[k], r.ID)
	}
	claim := func(kind string, hash []byte) (uuid.UUID, bool) {
		k := kind + string(hash)
		if ids := pool[k]; len(ids) > 0 {
			pool[k] = ids[1:]
			return ids[0], true
		}
		return uuid.Must(uuid.NewV7()), false
	}

	var p rowPlan
	for pi, par := range parents {
		hash := embedding.Hash(par.Heading + "\x00" + par.Content)
		id, _ := claim("parent", hash)
		p.rows = append(p.rows, store.Chunk{
			ID: id, Kind: "parent", Position: pi, Heading: par.Heading, Content: par.Content,
			Tokens: par.Tokens, ContentHash: hash,
		})
		p.tokens += par.Tokens
	}
	for pi, par := range parents {
		pid := p.rows[pi].ID
		for ci, c := range par.Children {
			var ctxText string
			if contexts != nil {
				ctxText = contexts[pi][ci]
			}
			input := EmbedInput(title, c.Heading, ctxText, c.Content)
			hash := embedding.Hash(input)
			id, reused := claim("child", hash)
			if reused {
				p.reusedChildren++
			} else {
				p.changed = append(p.changed, len(p.inputs))
			}
			p.childRow = append(p.childRow, len(p.rows))
			p.rows = append(p.rows, store.Chunk{
				ID: id, ParentID: &pid, Kind: "child", Position: len(p.inputs),
				Heading: c.Heading, Content: c.Content, Context: ctxText, Tokens: c.Tokens, ContentHash: hash,
			})
			p.inputs = append(p.inputs, input)
		}
	}
	for _, r := range old {
		if r.Kind != "child" {
			continue
		}
		for _, id := range pool["child"+string(r.ContentHash)] {
			if id == r.ID {
				p.obsolete = append(p.obsolete, id)
			}
		}
	}
	return p
}

// EmbedInput is the text embedded and keyword-indexed for a child: its
// document title and heading path, the LLM-written context if any, then the
// chunk itself — so "the limit is ten days" is indexed as being about the
// leave policy of that handbook.
func EmbedInput(title, heading, context, content string) string {
	var b strings.Builder
	b.WriteString(title)
	if heading != "" {
		b.WriteString(" > ")
		b.WriteString(heading)
	}
	b.WriteString("\n")
	if context != "" {
		b.WriteString(context)
		b.WriteString("\n")
	}
	b.WriteString("\n")
	b.WriteString(content)
	return b.String()
}
