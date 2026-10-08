// Package contextual implements Contextual Retrieval: before a chunk is
// embedded, an LLM writes one or two sentences placing it in its document
// ("this section of the 2024 employee handbook covers annual leave"), so a
// chunk that says only "the limit is ten days" can still be found by a
// question about leave.
package contextual

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"golang.org/x/sync/errgroup"

	"rag/internal/chunk"
	"rag/internal/llm"
)

type Generator struct {
	llm          *llm.Client
	db           *sql.DB
	tok          chunk.Tokenizer
	maxDocTokens int
	concurrency  int
}

func New(client *llm.Client, sqlDB *sql.DB, tok chunk.Tokenizer, maxDocTokens, concurrency int) *Generator {
	return &Generator{llm: client, db: sqlDB, tok: tok, maxDocTokens: maxDocTokens, concurrency: max(1, concurrency)}
}

// Item is one child chunk and the parent it belongs to.
type Item struct {
	Parent string
	Child  string
}

// Contexts returns one context per item, in order. Cached contexts are
// reused; the rest are generated with bounded concurrency.
func (g *Generator) Contexts(ctx context.Context, title, document string, items []Item) ([]string, error) {
	out := make([]string, len(items))
	keys := make([][]byte, len(items))
	for i, it := range items {
		// The prompt shows the document head and the parent, so the cache key
		// covers the title, parent and child: editing one section does not
		// invalidate the contexts of every other section.
		h := sha256.Sum256([]byte(title + "\x00" + it.Parent + "\x00" + it.Child))
		keys[i] = h[:]
	}
	if err := g.lookup(ctx, keys, out); err != nil {
		return nil, err
	}

	head := g.documentHead(document)
	eg, ectx := errgroup.WithContext(ctx)
	eg.SetLimit(g.concurrency)
	for i, it := range items {
		if out[i] != "" {
			continue
		}
		eg.Go(func() error {
			text, err := g.llm.Complete(ectx, prompt(title, head, it), 200, 3)
			if err != nil {
				return fmt.Errorf("chunk context: %w", err)
			}
			text = strings.TrimSpace(text)
			if text == "" {
				return errors.New("chunk context: LLM returned an empty answer")
			}
			out[i] = text
			_, err = g.db.ExecContext(ectx, `INSERT IGNORE INTO context_cache (model, hash, context) VALUES (?, ?, ?)`,
				g.llm.Model(), keys[i], text)
			return err
		})
	}
	if err := eg.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

func (g *Generator) lookup(ctx context.Context, keys [][]byte, out []string) error {
	byKey := map[string][]int{}
	for i, k := range keys {
		byKey[string(k)] = append(byKey[string(k)], i)
	}
	const batch = 1000
	for start := 0; start < len(keys); start += batch {
		part := keys[start:min(start+batch, len(keys))]
		args := []any{g.llm.Model()}
		for _, k := range part {
			args = append(args, k)
		}
		rows, err := g.db.QueryContext(ctx, `SELECT hash, context FROM context_cache WHERE model = ? AND hash IN (?`+
			strings.Repeat(",?", len(part)-1)+`)`, args...)
		if err != nil {
			return fmt.Errorf("context cache lookup: %w", err)
		}
		for rows.Next() {
			var k []byte
			var text string
			if err := rows.Scan(&k, &text); err != nil {
				rows.Close()
				return err
			}
			for _, i := range byKey[string(k)] {
				out[i] = text
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return err
		}
	}
	return nil
}

// documentHead returns the document, cut to maxDocTokens on a rune
// boundary. The parent section is always added to the prompt separately,
// so a chunk deep in a long document still gets its local surroundings.
func (g *Generator) documentHead(doc string) string {
	if g.tok.Count(doc) <= g.maxDocTokens {
		return doc
	}
	runes := []rune(doc)
	lo, hi := 0, len(runes)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if g.tok.Count(string(runes[:mid])) <= g.maxDocTokens {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return string(runes[:lo]) + "\n…（后文省略）"
}

func prompt(title, head string, it Item) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<document title=%q>\n%s\n</document>\n\n", title, head)
	fmt.Fprintf(&b, "<section>\n%s\n</section>\n\n", it.Parent)
	fmt.Fprintf(&b, "<chunk>\n%s\n</chunk>\n\n", it.Child)
	b.WriteString("上面的 <chunk> 是 <section> 中的一个片段，<section> 属于 <document>。" +
		"请用一两句话简要说明这个片段在整篇文档中的位置和主题（例如属于哪份文档、哪个章节，讲的是什么对象），" +
		"以便检索时能找到它。使用与文档相同的语言。只输出这句说明，不要复述片段内容，不要输出其他任何内容。")
	return b.String()
}
