// Package store holds the SQL for datasets, documents and chunks (MySQL).
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"rag/internal/db"
)

var ErrNotFound = errors.New("not found")

type ChunkConfig struct {
	ParentTokens int  `json:"parent_tokens,omitempty"`
	ChildTokens  int  `json:"child_tokens,omitempty"`
	Contextual   bool `json:"contextual,omitempty"` // Contextual Retrieval (needs an LLM)
}

type Dataset struct {
	ID             uuid.UUID   `json:"id"`
	TenantID       uuid.UUID   `json:"-"`
	Name           string      `json:"name"`
	Description    string      `json:"description"`
	EmbeddingModel string      `json:"embedding_model"`
	ChunkConfig    ChunkConfig `json:"chunk_config"`
	DocumentCount  int         `json:"document_count"`
	CreatedAt      time.Time   `json:"created_at"`
	UpdatedAt      time.Time   `json:"updated_at"`
}

type Document struct {
	ID          uuid.UUID      `json:"id"`
	TenantID    uuid.UUID      `json:"-"`
	DatasetID   uuid.UUID      `json:"dataset_id"`
	Name        string         `json:"name"`
	Format      string         `json:"format"`
	SizeBytes   int64          `json:"size_bytes"`
	ContentHash []byte         `json:"-"`
	Metadata    map[string]any `json:"metadata"`
	Status      string         `json:"status"`
	Stage       string         `json:"stage,omitempty"`
	Error       string         `json:"error,omitempty"`
	ChunkCount  int            `json:"chunk_count"`
	TokenCount  int            `json:"token_count"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	IndexedAt   *time.Time     `json:"indexed_at,omitempty"`
}

type Chunk struct {
	ID          uuid.UUID
	ParentID    *uuid.UUID
	Kind        string // parent | child
	Position    int
	Heading     string
	Content     string
	Context     string // LLM-written situating sentence (children, contextual datasets only)
	Tokens      int
	ContentHash []byte
}

// ChunkRef identifies an existing chunk by what it was built from.
type ChunkRef struct {
	ID          uuid.UUID
	Kind        string
	ContentHash []byte
}

type Store struct{ db *sql.DB }

func New(sqlDB *sql.DB) *Store { return &Store{db: sqlDB} }

func (s *Store) DB() *sql.DB { return s.db }

type scanner interface{ Scan(dest ...any) error }

// jsonCol scans a JSON column into v.
type jsonCol struct{ v any }

func (j jsonCol) Scan(src any) error {
	switch b := src.(type) {
	case []byte:
		return json.Unmarshal(b, j.v)
	case string:
		return json.Unmarshal([]byte(b), j.v)
	case nil:
		return nil
	}
	return fmt.Errorf("unexpected JSON column type %T", src)
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// ---- datasets ----

func (s *Store) CreateDataset(ctx context.Context, tx *sql.Tx, d *Dataset) error {
	now := time.Now().UTC().Truncate(time.Microsecond)
	_, err := tx.ExecContext(ctx, `INSERT INTO datasets
		(id, tenant_id, name, description, embedding_model, chunk_config, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		d.ID.String(), d.TenantID.String(), d.Name, d.Description, d.EmbeddingModel, mustJSON(d.ChunkConfig), now, now)
	d.CreatedAt, d.UpdatedAt = now, now
	return err
}

const datasetCols = `d.id, d.tenant_id, d.name, d.description, d.embedding_model, d.chunk_config,
	(SELECT COUNT(*) FROM documents x WHERE x.dataset_id = d.id), d.created_at, d.updated_at`

func scanDataset(row scanner) (*Dataset, error) {
	var d Dataset
	var id, tenant string
	err := row.Scan(&id, &tenant, &d.Name, &d.Description, &d.EmbeddingModel, jsonCol{&d.ChunkConfig},
		&d.DocumentCount, &d.CreatedAt, &d.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	d.ID, d.TenantID = uuid.MustParse(id), uuid.MustParse(tenant)
	return &d, nil
}

func (s *Store) GetDataset(ctx context.Context, tenant, id uuid.UUID) (*Dataset, error) {
	return scanDataset(s.db.QueryRowContext(ctx,
		`SELECT `+datasetCols+` FROM datasets d WHERE d.tenant_id = ? AND d.id = ?`, tenant.String(), id.String()))
}

func (s *Store) ListDatasets(ctx context.Context, tenant uuid.UUID) ([]*Dataset, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT `+datasetCols+` FROM datasets d WHERE d.tenant_id = ? ORDER BY d.created_at DESC`, tenant.String())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Dataset{}
	for rows.Next() {
		d, err := scanDataset(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// DeleteDataset removes the dataset (documents and chunks cascade) inside
// tx, so the caller can enqueue the vector cleanup in the same transaction.
func (s *Store) DeleteDataset(ctx context.Context, tx *sql.Tx, tenant, id uuid.UUID) error {
	res, err := tx.ExecContext(ctx, `DELETE FROM datasets WHERE tenant_id = ? AND id = ?`, tenant.String(), id.String())
	return affected(res, err)
}

func affected(res sql.Result, err error) error {
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}

// ---- documents ----

const documentCols = `id, tenant_id, dataset_id, name, format, size_bytes, content_hash, metadata,
	status, stage, error, chunk_count, token_count, created_at, updated_at, indexed_at`

func scanDocument(row scanner) (*Document, error) {
	var d Document
	var id, tenant, ds string
	var indexed sql.NullTime
	err := row.Scan(&id, &tenant, &ds, &d.Name, &d.Format, &d.SizeBytes, &d.ContentHash, jsonCol{&d.Metadata},
		&d.Status, &d.Stage, &d.Error, &d.ChunkCount, &d.TokenCount, &d.CreatedAt, &d.UpdatedAt, &indexed)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	d.ID, d.TenantID, d.DatasetID = uuid.MustParse(id), uuid.MustParse(tenant), uuid.MustParse(ds)
	if indexed.Valid {
		d.IndexedAt = &indexed.Time
	}
	if d.Metadata == nil {
		d.Metadata = map[string]any{}
	}
	return &d, nil
}

// InsertDocument writes the document row and its bytes inside tx, so the
// caller can enqueue the ingest job in the same transaction.
func (s *Store) InsertDocument(ctx context.Context, tx *sql.Tx, d *Document, data []byte) error {
	if d.Metadata == nil {
		d.Metadata = map[string]any{}
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	d.Status, d.CreatedAt, d.UpdatedAt = "queued", now, now
	if _, err := tx.ExecContext(ctx, `INSERT INTO documents
		(id, tenant_id, dataset_id, name, format, size_bytes, content_hash, metadata, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		d.ID.String(), d.TenantID.String(), d.DatasetID.String(), d.Name, d.Format, d.SizeBytes, d.ContentHash,
		mustJSON(d.Metadata), now, now); err != nil {
		return err
	}
	_, err := tx.ExecContext(ctx, `INSERT INTO document_blobs (document_id, data) VALUES (?, ?)`, d.ID.String(), data)
	return err
}

func (s *Store) FindDocumentByHash(ctx context.Context, datasetID uuid.UUID, hash []byte) (*Document, error) {
	return scanDocument(s.db.QueryRowContext(ctx,
		`SELECT `+documentCols+` FROM documents WHERE dataset_id = ? AND content_hash = ?`, datasetID.String(), hash))
}

func (s *Store) GetDocument(ctx context.Context, tenant, id uuid.UUID) (*Document, error) {
	return scanDocument(s.db.QueryRowContext(ctx,
		`SELECT `+documentCols+` FROM documents WHERE tenant_id = ? AND id = ?`, tenant.String(), id.String()))
}

func (s *Store) ListDocuments(ctx context.Context, tenant, datasetID uuid.UUID) ([]*Document, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT `+documentCols+` FROM documents
		WHERE tenant_id = ? AND dataset_id = ? ORDER BY created_at DESC`, tenant.String(), datasetID.String())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Document{}
	for rows.Next() {
		d, err := scanDocument(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// DeleteDocument removes the document (chunks and blob cascade) inside tx.
func (s *Store) DeleteDocument(ctx context.Context, tx *sql.Tx, tenant, id uuid.UUID) error {
	res, err := tx.ExecContext(ctx, `DELETE FROM documents WHERE tenant_id = ? AND id = ?`, tenant.String(), id.String())
	return affected(res, err)
}

// MarkQueued resets a document before it is re-enqueued, inside the
// enqueueing transaction.
func (s *Store) MarkQueued(ctx context.Context, tx *sql.Tx, id uuid.UUID) error {
	_, err := tx.ExecContext(ctx, `UPDATE documents SET status = 'queued', stage = '', error = '', updated_at = NOW(6)
		WHERE id = ?`, id.String())
	return err
}

// LoadForIngest returns everything the indexing pipeline needs for one document.
func (s *Store) LoadForIngest(ctx context.Context, id uuid.UUID) (*Document, *Dataset, []byte, error) {
	doc, err := scanDocument(s.db.QueryRowContext(ctx, `SELECT `+documentCols+` FROM documents WHERE id = ?`, id.String()))
	if err != nil {
		return nil, nil, nil, err
	}
	ds, err := s.GetDataset(ctx, doc.TenantID, doc.DatasetID)
	if err != nil {
		return nil, nil, nil, err
	}
	var data []byte
	err = s.db.QueryRowContext(ctx, `SELECT data FROM document_blobs WHERE document_id = ?`, id.String()).Scan(&data)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, nil, ErrNotFound
	}
	return doc, ds, data, err
}

func (s *Store) SetProgress(ctx context.Context, id uuid.UUID, status, stage, errMsg string) error {
	_, err := s.db.ExecContext(ctx, `UPDATE documents SET status = ?, stage = ?, error = ?, updated_at = NOW(6)
		WHERE id = ?`, status, stage, errMsg, id.String())
	return err
}

// ChunkRefs lists the document's current chunks (children's ids are also
// their vector ids), for matching against a re-chunked version.
func (s *Store) ChunkRefs(ctx context.Context, documentID uuid.UUID) ([]ChunkRef, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id, kind, content_hash FROM chunks WHERE document_id = ?`, documentID.String())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ChunkRef
	for rows.Next() {
		var r ChunkRef
		var id string
		if err := rows.Scan(&id, &r.Kind, &r.ContentHash); err != nil {
			return nil, err
		}
		r.ID = uuid.MustParse(id)
		out = append(out, r)
	}
	return out, rows.Err()
}

// UpdateDocumentMetadata replaces a document's metadata inside tx.
func (s *Store) UpdateDocumentMetadata(ctx context.Context, tx *sql.Tx, tenant, id uuid.UUID, metadata map[string]any) error {
	res, err := tx.ExecContext(ctx, `UPDATE documents SET metadata = ?, updated_at = NOW(6) WHERE tenant_id = ? AND id = ?`,
		mustJSON(metadata), tenant.String(), id.String())
	if err != nil {
		return err
	}
	// RowsAffected is 0 both for "no such row" and "same value"; check existence explicitly.
	if n, _ := res.RowsAffected(); n == 0 {
		var one int
		if err := tx.QueryRowContext(ctx, `SELECT 1 FROM documents WHERE tenant_id = ? AND id = ?`, tenant.String(), id.String()).Scan(&one); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return ErrNotFound
			}
			return err
		}
	}
	return nil
}

// ReplaceDocumentFile swaps in a new version of the document's file inside
// tx; the document keeps its id, so its chunks can be updated incrementally.
func (s *Store) ReplaceDocumentFile(ctx context.Context, tx *sql.Tx, tenant, id uuid.UUID, name, format string, hash, data []byte) error {
	res, err := tx.ExecContext(ctx, `UPDATE documents SET name = ?, format = ?, size_bytes = ?, content_hash = ?, updated_at = NOW(6)
		WHERE tenant_id = ? AND id = ?`, name, format, len(data), hash, tenant.String(), id.String())
	if err := affected(res, err); err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx, `UPDATE document_blobs SET data = ? WHERE document_id = ?`, data, id.String())
	return err
}

// ReplaceChunks swaps a document's chunks atomically: readers see either the
// old set or the new one. afterwards runs inside the same transaction (used
// to enqueue deletion of the old vectors).
func (s *Store) ReplaceChunks(ctx context.Context, doc *Document, chunks []Chunk, tokenCount int, afterwards func(*sql.Tx) error) error {
	return db.Tx(ctx, s.db, func(tx *sql.Tx) error {
		// Lock the row: if the document was deleted meanwhile, stop here.
		var one int
		err := tx.QueryRowContext(ctx, `SELECT 1 FROM documents WHERE id = ? FOR UPDATE`, doc.ID.String()).Scan(&one)
		if errors.Is(err, sql.ErrNoRows) {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM chunks WHERE document_id = ?`, doc.ID.String()); err != nil {
			return err
		}
		const cols = 12
		const batch = 500
		for start := 0; start < len(chunks); start += batch {
			end := min(start+batch, len(chunks))
			var sb strings.Builder
			sb.WriteString(`INSERT INTO chunks (id, tenant_id, dataset_id, document_id, parent_id, kind, position,
				heading, content, context, tokens, content_hash) VALUES `)
			args := make([]any, 0, (end-start)*cols)
			for i, c := range chunks[start:end] {
				if i > 0 {
					sb.WriteString(",")
				}
				sb.WriteString("(?,?,?,?,?,?,?,?,?,?,?,?)")
				var parent, context any
				if c.ParentID != nil {
					parent = c.ParentID.String()
				}
				if c.Context != "" {
					context = c.Context
				}
				args = append(args, c.ID.String(), doc.TenantID.String(), doc.DatasetID.String(), doc.ID.String(),
					parent, c.Kind, c.Position, c.Heading, c.Content, context, c.Tokens, c.ContentHash)
			}
			if _, err := tx.ExecContext(ctx, sb.String(), args...); err != nil {
				return fmt.Errorf("insert chunks: %w", err)
			}
		}
		if _, err := tx.ExecContext(ctx, `UPDATE documents SET status = 'completed', stage = '', error = '',
			chunk_count = ?, token_count = ?, indexed_at = NOW(6), updated_at = NOW(6) WHERE id = ?`,
			len(chunks), tokenCount, doc.ID.String()); err != nil {
			return err
		}
		if afterwards != nil {
			return afterwards(tx)
		}
		return nil
	})
}

// ---- retrieval ----

type ChunkView struct {
	ChunkID      uuid.UUID
	ParentID     uuid.UUID // == ChunkID for a parentless chunk
	DocumentID   uuid.UUID
	DocumentName string
	Heading      string
	Content      string // parent content when there is a parent
	Metadata     map[string]any
}

// ChunkViews loads enabled chunks by id with their parent's content and
// their document. Ids that no longer exist are absent from the result.
func (s *Store) ChunkViews(ctx context.Context, tenant uuid.UUID, ids []uuid.UUID) (map[uuid.UUID]*ChunkView, error) {
	out := make(map[uuid.UUID]*ChunkView, len(ids))
	if len(ids) == 0 {
		return out, nil
	}
	args := []any{tenant.String()}
	for _, id := range ids {
		args = append(args, id.String())
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT c.id, COALESCE(c.parent_id, c.id), c.document_id, d.name,
		       COALESCE(p.heading, c.heading), COALESCE(p.content, c.content), d.metadata
		FROM chunks c
		JOIN documents d ON d.id = c.document_id
		LEFT JOIN chunks p ON p.id = c.parent_id
		WHERE c.tenant_id = ? AND c.enabled AND c.id IN (?`+strings.Repeat(",?", len(ids)-1)+`)`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var v ChunkView
		var id, parent, doc string
		if err := rows.Scan(&id, &parent, &doc, &v.DocumentName, &v.Heading, &v.Content, jsonCol{&v.Metadata}); err != nil {
			return nil, err
		}
		v.ChunkID, v.ParentID, v.DocumentID = uuid.MustParse(id), uuid.MustParse(parent), uuid.MustParse(doc)
		out[v.ChunkID] = &v
	}
	return out, rows.Err()
}

// DocumentsWhereMetadata returns ids of documents in the datasets whose
// metadata field, read as text, matches a LIKE pattern. Used for filter
// operators the vector store cannot evaluate (prefix / suffix).
func (s *Store) DocumentsWhereMetadata(ctx context.Context, tenant uuid.UUID, datasets []uuid.UUID, field, likePattern string) ([]string, error) {
	if len(datasets) == 0 {
		return nil, nil
	}
	args := []any{tenant.String()}
	for _, d := range datasets {
		args = append(args, d.String())
	}
	args = append(args, JSONPath(field), likePattern)
	rows, err := s.db.QueryContext(ctx, `SELECT id FROM documents
		WHERE tenant_id = ? AND dataset_id IN (?`+strings.Repeat(",?", len(datasets)-1)+`)
		AND JSON_UNQUOTE(JSON_EXTRACT(metadata, ?)) LIKE ? ESCAPE '!'`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// JSONPath quotes a top-level key as a MySQL JSON path: $."key".
func JSONPath(key string) string {
	return `$."` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(key) + `"`
}

// EscapeLike escapes LIKE wildcards for ESCAPE '!'.
func EscapeLike(s string) string {
	return strings.NewReplacer("!", "!!", "%", "!%", "_", "!_").Replace(s)
}

// ---- quotas ----

// Limits of a tenant; 0 means unlimited.
type Limits struct {
	MaxDatasets  int
	MaxDocuments int
	MaxStorageMB int
}

// QuotaError: the change would exceed a tenant limit.
type QuotaError struct{ Msg string }

func (e *QuotaError) Error() string { return e.Msg }

type Usage struct {
	Datasets     int   `json:"datasets"`
	Documents    int   `json:"documents"`
	StorageBytes int64 `json:"storage_bytes"`
	Chunks       int   `json:"chunks"`
}

func (s *Store) Usage(ctx context.Context, q interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}, tenant uuid.UUID) (*Usage, error) {
	var u Usage
	err := q.QueryRowContext(ctx, `SELECT
		(SELECT COUNT(*) FROM datasets WHERE tenant_id = ?),
		(SELECT COUNT(*) FROM documents WHERE tenant_id = ?),
		(SELECT COALESCE(SUM(size_bytes), 0) FROM documents WHERE tenant_id = ?),
		(SELECT COUNT(*) FROM chunks WHERE tenant_id = ?)`,
		tenant.String(), tenant.String(), tenant.String(), tenant.String()).
		Scan(&u.Datasets, &u.Documents, &u.StorageBytes, &u.Chunks)
	return &u, err
}

// CheckQuota verifies, inside tx, that adding the given amounts keeps the
// tenant within its limits. It locks the tenant row, so concurrent uploads
// of one tenant are checked one after another and cannot overshoot together.
func (s *Store) CheckQuota(ctx context.Context, tx *sql.Tx, tenant uuid.UUID, lim Limits, addDatasets, addDocuments int, addBytes int64) error {
	if lim.MaxDatasets == 0 && lim.MaxDocuments == 0 && lim.MaxStorageMB == 0 {
		return nil
	}
	var one int
	if err := tx.QueryRowContext(ctx, `SELECT 1 FROM tenants WHERE id = ? FOR UPDATE`, tenant.String()).Scan(&one); err != nil {
		return err
	}
	u, err := s.Usage(ctx, tx, tenant)
	if err != nil {
		return err
	}
	switch {
	case lim.MaxDatasets > 0 && u.Datasets+addDatasets > lim.MaxDatasets:
		return &QuotaError{fmt.Sprintf("dataset quota reached (%d)", lim.MaxDatasets)}
	case lim.MaxDocuments > 0 && u.Documents+addDocuments > lim.MaxDocuments:
		return &QuotaError{fmt.Sprintf("document quota reached (%d)", lim.MaxDocuments)}
	case lim.MaxStorageMB > 0 && u.StorageBytes+addBytes > int64(lim.MaxStorageMB)<<20:
		return &QuotaError{fmt.Sprintf("storage quota reached (%d MB)", lim.MaxStorageMB)}
	}
	return nil
}
