package embedding

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// Hash is the cache key and chunks.content_hash for an embedding input.
func Hash(text string) []byte {
	h := sha256.Sum256([]byte(text))
	return h[:]
}

// Cached wraps a Client with a (model, sha256) → vector table. Lookups for a
// whole document take one query per 1000 inputs; only misses go to the model.
type Cached struct {
	db     *sql.DB
	client *Client
}

func NewCached(sqlDB *sql.DB, client *Client) *Cached {
	return &Cached{db: sqlDB, client: client}
}

func (c *Cached) Model() string { return c.client.Model() }

const cacheBatch = 1000

// Embed returns one vector per input, in input order.
func (c *Cached) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	byHash := map[string][]int{} // identical inputs share one lookup and one model call
	hashes := make([][]byte, 0, len(texts))
	for i, t := range texts {
		h := Hash(t)
		if _, seen := byHash[string(h)]; !seen {
			hashes = append(hashes, h)
		}
		byHash[string(h)] = append(byHash[string(h)], i)
	}

	for start := 0; start < len(hashes); start += cacheBatch {
		part := hashes[start:min(start+cacheBatch, len(hashes))]
		args := []any{c.client.Model()}
		for _, h := range part {
			args = append(args, h)
		}
		rows, err := c.db.QueryContext(ctx, `SELECT hash, embedding FROM embedding_cache
			WHERE model = ? AND hash IN (?`+strings.Repeat(",?", len(part)-1)+`)`, args...)
		if err != nil {
			return nil, fmt.Errorf("embedding cache lookup: %w", err)
		}
		for rows.Next() {
			var h, blob []byte
			if err := rows.Scan(&h, &blob); err != nil {
				rows.Close()
				return nil, err
			}
			vec, ok := decode(blob)
			if !ok || Validate(vec, c.client.cfg.Dim) != nil {
				continue // stale or corrupt entry: re-embed
			}
			for _, i := range byHash[string(h)] {
				out[i] = vec
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, fmt.Errorf("embedding cache lookup: %w", err)
		}
	}

	var missTexts []string
	var missHashes [][]byte
	for _, h := range hashes {
		idx := byHash[string(h)]
		if out[idx[0]] == nil {
			missTexts = append(missTexts, texts[idx[0]])
			missHashes = append(missHashes, h)
		}
	}
	if len(missTexts) == 0 {
		return out, nil
	}

	vecs, err := c.client.Embed(ctx, missTexts)
	if err != nil {
		return nil, err
	}
	for i, v := range vecs {
		for _, j := range byHash[string(missHashes[i])] {
			out[j] = v
		}
	}
	// ~4 KB per 1024-dim row; 200 rows keeps a statement well under max_allowed_packet.
	const writeBatch = 200
	for start := 0; start < len(vecs); start += writeBatch {
		end := min(start+writeBatch, len(vecs))
		args := make([]any, 0, (end-start)*3)
		for i := start; i < end; i++ {
			args = append(args, c.client.Model(), missHashes[i], encode(vecs[i]))
		}
		_, err := c.db.ExecContext(ctx, `INSERT IGNORE INTO embedding_cache (model, hash, embedding) VALUES (?,?,?)`+
			strings.Repeat(",(?,?,?)", end-start-1), args...)
		if err != nil {
			return nil, fmt.Errorf("embedding cache write: %w", err)
		}
	}
	return out, nil
}

func encode(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func decode(b []byte) ([]float32, bool) {
	if len(b)%4 != 0 {
		return nil, false
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, true
}
