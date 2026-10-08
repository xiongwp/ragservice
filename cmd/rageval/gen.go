package main

// rageval gen: build a question set from a folder of documents.
//
// Passages are cut from the parsed text with a plain paragraph window,
// independent of how either system chunks, so neither side is favoured.
// For each sampled passage an LLM writes a question a colleague might ask
// and copies a short answer phrase verbatim from the passage; a phrase that
// is not actually in the passage, or a question that just repeats it, is
// discarded. Questions carry the passage so a human can spot-check them.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"rag/internal/chunk"
	"rag/internal/config"
	"rag/internal/llm"
	"rag/internal/parse"
)

type passage struct {
	doc  string
	text string
}

type generated struct {
	ID         string   `json:"id"`
	Query      string   `json:"query"`
	Expect     []string `json:"expect"`
	ExpectDocs []string `json:"expect_docs"`
	Passage    string   `json:"passage"` // for spot-checking; ignored by run
}

func runGen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	docs := fs.String("docs", "", "folder of documents")
	n := fs.Int("n", 150, "number of questions")
	out := fs.String("out", "eval/questions.jsonl", "output JSONL")
	docling := fs.String("docling", "http://localhost:5001", "docling-serve URL for pdf/docx/pptx/xlsx")
	llmURL := fs.String("llm-url", "https://api.siliconflow.cn/v1", "OpenAI-compatible base URL")
	llmModel := fs.String("llm-model", "Qwen/Qwen2.5-72B-Instruct", "model that writes the questions")
	seed := fs.Uint64("seed", 42, "sampling seed (same seed, same passages)")
	conc := fs.Int("concurrency", 4, "LLM calls in flight")
	cacheDir := fs.String("cache", "eval/.parsed", "keep parsed text here so re-runs skip docling")
	fs.Parse(args)
	key := firstEnv("LLM_API_KEY", "EMBEDDING_API_KEY")
	if *docs == "" || key == "" {
		fatal(fmt.Errorf("gen needs -docs and an API key in LLM_API_KEY or EMBEDDING_API_KEY"))
	}

	passages, err := collectPassages(*docs, *docling, *cacheDir)
	if err != nil {
		fatal(err)
	}
	if len(passages) == 0 {
		fatal(fmt.Errorf("no usable text found in %s", *docs))
	}
	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))
	rng.Shuffle(len(passages), func(i, j int) { passages[i], passages[j] = passages[j], passages[i] })
	pool := spread(passages)
	fmt.Fprintf(os.Stderr, "%d candidate passages from %d documents\n", len(passages), countDocs(passages))

	client := llm.New(config.LLMConfig{BaseURL: strings.TrimRight(*llmURL, "/"), APIKey: key, Model: *llmModel,
		Timeout: 90 * time.Second, Concurrency: *conc, RPS: 5})
	var (
		mu      sync.Mutex
		results []generated
		wg      sync.WaitGroup
		next    int
		tried   int
		dropped int
		failed  int
		stop    bool // the LLM refused for good (bad key, no balance): stop asking
	)
	sem := make(chan struct{}, *conc)
	for {
		mu.Lock()
		done := len(results) >= *n || next >= len(pool) || stop
		mu.Unlock()
		if done {
			break
		}
		p := pool[next]
		next++
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			q, ok, err := askQuestion(client, p)
			mu.Lock()
			defer mu.Unlock()
			tried++
			if err != nil {
				failed++
				if errors.Is(err, llm.ErrPermanent) && !stop {
					stop = true
					fmt.Fprintf(os.Stderr, "  LLM refused the request, stopping: %v\n", err)
				}
				return
			}
			if !ok {
				dropped++
				return
			}
			if len(results) < *n {
				q.ID = fmt.Sprintf("q%03d", len(results)+1)
				results = append(results, q)
				if len(results)%10 == 0 {
					fmt.Fprintf(os.Stderr, "  %d questions\n", len(results))
				}
			}
		}()
	}
	wg.Wait()

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		fatal(err)
	}
	f, err := os.Create(*out)
	if err != nil {
		fatal(err)
	}
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	for _, q := range results {
		enc.Encode(q)
	}
	f.Close()
	fmt.Fprintf(os.Stderr, "wrote %d questions to %s (%d passages tried: %d discarded by the checks, %d LLM calls failed)\n",
		len(results), *out, tried, dropped, failed)
	if len(results) < *n {
		fmt.Fprintf(os.Stderr, "note: asked for %d; the documents only yielded %d valid questions\n", *n, len(results))
	}
}

func firstEnv(names ...string) string {
	for _, n := range names {
		if v := os.Getenv(n); v != "" {
			return v
		}
	}
	return ""
}

func collectPassages(dir, doclingURL, cacheDir string) ([]passage, error) {
	parser := parse.New(doclingURL, time.Hour, 1)
	var out []passage
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || strings.HasPrefix(d.Name(), ".") {
			return err
		}
		format := parse.FormatOf(d.Name())
		if format == "" {
			fmt.Fprintf(os.Stderr, "skip %s (unsupported type)\n", d.Name())
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(data)
		cached := filepath.Join(cacheDir, hex.EncodeToString(sum[:16])+".md")
		var md string
		if b, err := os.ReadFile(cached); err == nil {
			md = string(b)
		} else {
			fmt.Fprintf(os.Stderr, "parsing %s...\n", d.Name())
			md, err = parser.Parse(context.Background(), d.Name(), format, data)
			if err != nil {
				fmt.Fprintf(os.Stderr, "skip %s: %v\n", d.Name(), err)
				return nil
			}
			if os.MkdirAll(cacheDir, 0o755) == nil {
				_ = os.WriteFile(cached, []byte(md), 0o644)
			}
		}
		for _, t := range windows(chunk.Clean(md)) {
			out = append(out, passage{doc: d.Name(), text: t})
		}
		return nil
	})
	return out, err
}

// windows groups paragraphs into passages of roughly 150–600 characters.
func windows(text string) []string {
	var out []string
	var cur []string
	size := 0
	flush := func() {
		t := strings.TrimSpace(strings.Join(cur, "\n\n"))
		if utf8.RuneCountInString(t) >= 120 && !onlyHeadings(cur) {
			out = append(out, t)
		}
		cur, size = nil, 0
	}
	for _, para := range strings.Split(text, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		l := utf8.RuneCountInString(para)
		if size > 0 && size+l > 600 {
			flush()
		}
		cur = append(cur, para)
		size += l
		if size >= 150 && !strings.HasPrefix(para, "#") {
			flush()
		}
	}
	flush()
	return out
}

func onlyHeadings(paras []string) bool {
	for _, p := range paras {
		if !strings.HasPrefix(p, "#") {
			return false
		}
	}
	return true
}

// spread orders passages round-robin across documents, so a few long
// documents cannot take every question.
func spread(ps []passage) []passage {
	byDoc := map[string][]passage{}
	var docs []string
	for _, p := range ps {
		if _, ok := byDoc[p.doc]; !ok {
			docs = append(docs, p.doc)
		}
		byDoc[p.doc] = append(byDoc[p.doc], p)
	}
	sort.Strings(docs)
	var out []passage
	for i := 0; len(out) < len(ps); i++ {
		for _, d := range docs {
			if i < len(byDoc[d]) {
				out = append(out, byDoc[d][i])
			}
		}
	}
	return out
}

func countDocs(ps []passage) int {
	seen := map[string]bool{}
	for _, p := range ps {
		seen[p.doc] = true
	}
	return len(seen)
}

const genPrompt = `你在为知识库检索系统准备测试题。根据下面这段文档内容出一道题。

要求：
1. 问题要像读者真实会问的那样，用自己的话提问，不要照抄段落里的句子。问题使用与段落主要内容相同的语言（英文段落用英文提问，中文段落用中文提问）。
2. 问题必须能仅凭这段内容回答，答案要具体（数字、规定、步骤、名称等），不要出"这段讲了什么"这类概括题。
3. answer 必须是从段落中**逐字复制**的一个连续片段（中文 8 到 40 个字，英文 3 到 12 个词），是回答问题的关键信息；不要跨越表格单元格，不要包含 | # * 等格式符号。
4. 如果这段内容没有适合出题的具体信息（例如只是目录、页眉、免责声明），question 和 answer 都输出空字符串。

文档：%s
<passage>
%s
</passage>

只输出 JSON：{"question": "...", "answer": "..."}`

func askQuestion(client *llm.Client, p passage) (generated, bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	out, err := client.Complete(ctx, fmt.Sprintf(genPrompt, p.doc, p.text), 300, 3)
	if err != nil {
		return generated{}, false, err
	}
	out = strings.TrimSpace(out)
	out = strings.TrimPrefix(strings.TrimPrefix(out, "```json"), "```")
	out = strings.TrimSpace(strings.TrimSuffix(out, "```"))
	if i, j := strings.Index(out, "{"), strings.LastIndex(out, "}"); i >= 0 && j > i {
		out = out[i : j+1]
	}
	var r struct {
		Question string `json:"question"`
		Answer   string `json:"answer"`
	}
	if json.Unmarshal([]byte(out), &r) != nil {
		return generated{}, false, nil
	}
	q, a := strings.TrimSpace(r.Question), strings.TrimSpace(r.Answer)
	n := utf8.RuneCountInString(a)
	switch {
	case q == "" || a == "" || n < 4 || n > 100:
		return generated{}, false, nil
	case strings.ContainsAny(a, "|#*`"): // formatting differs between extractors
		return generated{}, false, nil
	case !strings.Contains(normalize(p.text), normalize(a)): // must be verbatim
		return generated{}, false, nil
	case strings.Contains(normalize(q), normalize(a)): // the question gives the answer away
		return generated{}, false, nil
	}
	return generated{Query: q, Expect: []string{a}, ExpectDocs: []string{p.doc}, Passage: p.text}, true, nil
}
