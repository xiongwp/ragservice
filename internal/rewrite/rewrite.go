// Package rewrite expands a query with an LLM before retrieval:
//
//   - history: with conversation turns (only /v1/search has them; Dify's
//     /retrieval sends a bare query), the last question is first rewritten
//     into a standalone one ("它多少钱" → "企业版多少钱").
//   - multi: up to three paraphrases, each searched like the query itself.
//   - hyde: a short hypothetical answer passage, searched by vector only
//     (answers look more like documents than questions do).
//
// Rewriting is online, so it gets one attempt under a short deadline; on any
// failure retrieval proceeds with the original query.
package rewrite

import (
	"container/list"
	"context"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"

	"rag/internal/llm"
)

type Turn struct {
	Role    string `json:"role"` // user | assistant
	Content string `json:"content"`
}

type Expansion struct {
	Query    string   // standalone query (the original when there is no history)
	Queries  []string // paraphrases: vector + keyword search
	Passages []string // hypothetical answers: vector search only
}

type Rewriter struct {
	llm     *llm.Client
	mode    string // multi | hyde
	timeout time.Duration
	log     *slog.Logger
	cache   *lru
}

func New(client *llm.Client, mode string, log *slog.Logger) *Rewriter {
	return &Rewriter{llm: client, mode: mode, timeout: 8 * time.Second, log: log, cache: newLRU(2000)}
}

func (r *Rewriter) Expand(ctx context.Context, query string, history []Turn) Expansion {
	key := r.mode + "\x00" + query
	for _, t := range history {
		key += "\x00" + t.Role + ":" + t.Content
	}
	if e, ok := r.cache.get(key); ok {
		return e
	}
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	e := Expansion{Query: query}
	if len(history) > 0 {
		standalone, err := r.llm.Complete(ctx, standalonePrompt(query, history), 200, 1)
		if err != nil {
			r.log.Warn("query rewrite failed; using the original query", "step", "standalone", "err", err)
			return e
		}
		if s := firstLine(standalone); s != "" {
			e.Query = s
		}
	}
	switch r.mode {
	case "multi":
		out, err := r.llm.Complete(ctx, multiPrompt(e.Query), 300, 1)
		if err != nil {
			r.log.Warn("query rewrite failed; using the original query", "step", "multi", "err", err)
			return e
		}
		e.Queries = paraphrases(out, e.Query, 3)
	case "hyde":
		out, err := r.llm.Complete(ctx, hydePrompt(e.Query), 300, 1)
		if err != nil {
			r.log.Warn("query rewrite failed; using the original query", "step", "hyde", "err", err)
			return e
		}
		if out = strings.TrimSpace(out); out != "" {
			e.Passages = []string{out}
		}
	}
	r.cache.put(key, e)
	return e
}

func standalonePrompt(query string, history []Turn) string {
	var b strings.Builder
	b.WriteString("下面是一段对话历史和用户的最新问题。请把最新问题改写成一个不依赖对话历史、可以单独用于检索的完整问题" +
		"（补全代词和省略的主语、对象）。使用与问题相同的语言，只输出改写后的问题。\n\n<history>\n")
	for _, t := range history[max(0, len(history)-6):] { // the last few turns carry the referents
		fmt.Fprintf(&b, "%s: %s\n", t.Role, t.Content)
	}
	fmt.Fprintf(&b, "</history>\n\n最新问题：%s", query)
	return b.String()
}

func multiPrompt(query string) string {
	return "为了在知识库中检索到相关内容，请为下面的问题写出 3 个不同的表述方式（换用同义词、更具体或更概括的说法）。" +
		"使用与问题相同的语言，每行一个，不要编号，不要输出其他内容。\n\n问题：" + query
}

func hydePrompt(query string) string {
	return "请写一段 100 字左右的文字，像是从公司内部文档中摘录的、能够回答下面问题的段落。" +
		"不知道具体事实时可以合理假设，但要写成文档的口吻。使用与问题相同的语言，只输出这段文字。\n\n问题：" + query
}

var listMarker = regexp.MustCompile(`^\s*(?:[-*•]|\d+[.)、．）:：]|[（(]\d+[）)])\s*`)

func paraphrases(out, original string, n int) []string {
	seen := map[string]bool{strings.TrimSpace(original): true}
	var qs []string
	for _, line := range strings.Split(out, "\n") {
		q := strings.TrimSpace(listMarker.ReplaceAllString(line, ""))
		if q == "" || seen[q] {
			continue
		}
		seen[q] = true
		qs = append(qs, q)
		if len(qs) == n {
			break
		}
	}
	return qs
}

func firstLine(s string) string {
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			return strings.Trim(l, "\"“”")
		}
	}
	return ""
}

// lru is a small fixed-size cache: popular questions repeat, and a cached
// expansion saves a full LLM round trip.
type lru struct {
	mu    sync.Mutex
	cap   int
	order *list.List
	items map[string]*list.Element
}

type entry struct {
	key string
	val Expansion
}

func newLRU(capacity int) *lru {
	return &lru{cap: capacity, order: list.New(), items: map[string]*list.Element{}}
}

func (c *lru) get(key string) (Expansion, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.order.MoveToFront(el)
		return el.Value.(*entry).val, true
	}
	return Expansion{}, false
}

func (c *lru) put(key string, val Expansion) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		el.Value.(*entry).val = val
		c.order.MoveToFront(el)
		return
	}
	c.items[key] = c.order.PushFront(&entry{key, val})
	if c.order.Len() > c.cap {
		last := c.order.Back()
		c.order.Remove(last)
		delete(c.items, last.Value.(*entry).key)
	}
}
