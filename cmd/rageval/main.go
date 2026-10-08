// rageval measures retrieval quality on a labelled question set and compares
// systems side by side: ragd (or any Dify external-knowledge endpoint) and
// Dify's own knowledge base.
//
//	rageval gen  -docs eval/docs -n 150            # questions from your documents (gen.go)
//	rageval load -docs eval/docs -rag-url ... -dify-url ...   # same documents into both (load.go)
//	rageval run  -set questions.jsonl -k 5 \
//	  -target 'name=ragd,kind=external,url=http://localhost:8080,key=RAG_KEY,knowledge=<dataset_id>' \
//	  -target 'name=dify,kind=dify,url=http://localhost/v1,key=dataset-xxx,knowledge=<dify_dataset_id>' \
//	  -json report.json
//
// questions.jsonl, one object per line:
//
//	{"id": "q1", "query": "工龄满十年年假几天", "expect": ["年假为十天"]}
//	{"id": "q2", "query": "VPN 在哪下载", "expect_docs": ["IT 指南.pdf"]}
//
// A retrieved record is relevant if its content contains any "expect"
// phrase (whitespace and case ignored) or, when "expect" is empty, if its
// document is one of "expect_docs". Chunk ids differ between systems, so
// relevance is judged on content.
//
// For kind=dify, results use the knowledge base's own retrieval settings in
// Dify (set its Top K >= -k), unless -dify-retrieval-model overrides them.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
)

type question struct {
	ID         string   `json:"id"`
	Query      string   `json:"query"`
	Expect     []string `json:"expect"`
	ExpectDocs []string `json:"expect_docs"`
}

type record struct {
	Content string  `json:"content"`
	Title   string  `json:"title"`
	Score   float64 `json:"score"`
}

type target struct {
	Name, Kind, URL, Key, Knowledge string
	Retrieval                       json.RawMessage // kind=dify: retrieval_model to send
}

type targetFlags []target

func (t *targetFlags) String() string { return fmt.Sprint(*t) }
func (t *targetFlags) Set(v string) error {
	var tg target
	for _, kv := range strings.Split(v, ",") {
		k, val, ok := strings.Cut(kv, "=")
		if !ok {
			return fmt.Errorf("bad target field %q (want key=value)", kv)
		}
		switch strings.TrimSpace(k) {
		case "name":
			tg.Name = val
		case "kind":
			tg.Kind = val
		case "url":
			tg.URL = strings.TrimRight(val, "/")
		case "key":
			tg.Key = val
		case "knowledge":
			tg.Knowledge = val
		case "retrieval": // path to a JSON file with Dify's retrieval_model
			b, err := os.ReadFile(val)
			if err != nil {
				return err
			}
			if !json.Valid(b) {
				return fmt.Errorf("%s is not valid JSON", val)
			}
			tg.Retrieval = b
		default:
			return fmt.Errorf("unknown target field %q", k)
		}
	}
	if tg.Kind != "external" && tg.Kind != "dify" {
		return fmt.Errorf("target kind must be external or dify, got %q", tg.Kind)
	}
	if tg.URL == "" || tg.Knowledge == "" {
		return fmt.Errorf("target needs url and knowledge")
	}
	if tg.Name == "" {
		tg.Name = tg.Kind
	}
	*t = append(*t, tg)
	return nil
}

type outcome struct {
	QuestionID string   `json:"question_id"`
	Query      string   `json:"query"`
	Latency    float64  `json:"latency_ms"`
	Error      string   `json:"error,omitempty"`
	Recall     float64  `json:"recall"`
	Hit        float64  `json:"hit"`
	RR         float64  `json:"reciprocal_rank"`
	NDCG       float64  `json:"ndcg"`
	Relevant   []bool   `json:"relevant"`
	Titles     []string `json:"titles"`
	// Same metric over the first `budget` characters of the results, so a
	// system returning longer passages gets no free credit.
	RecallBudget float64 `json:"recall_at_budget"`
	Chars        int     `json:"returned_chars"`
}

type summary struct {
	Target   string    `json:"target"`
	K        int       `json:"k"`
	N        int       `json:"questions"`
	Errors   int       `json:"errors"`
	Recall   float64   `json:"recall_at_k"`
	Hit      float64   `json:"hit_at_k"`
	MRR      float64   `json:"mrr"`
	NDCG     float64   `json:"ndcg_at_k"`
	Budget   int       `json:"budget_chars"`
	RecallB  float64   `json:"recall_at_budget"`
	AvgChars float64   `json:"avg_returned_chars"`
	P50      float64   `json:"latency_p50_ms"`
	P95      float64   `json:"latency_p95_ms"`
	Outcomes []outcome `json:"outcomes"`
}

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "gen":
			runGen(os.Args[2:])
			return
		case "load":
			runLoad(os.Args[2:])
			return
		case "run":
			os.Args = append(os.Args[:1], os.Args[2:]...)
		}
	}
	var targets targetFlags
	setPath := flag.String("set", "", "question set (JSONL)")
	budget := flag.Int("budget", 2000, "characters of retrieved text scored by recall@budget")
	k := flag.Int("k", 5, "evaluate the top k records")
	conc := flag.Int("concurrency", 4, "queries in flight per target")
	jsonOut := flag.String("json", "", "write the full per-question report here")
	difyModel := flag.String("dify-retrieval-model", "", "JSON retrieval_model to send to Dify (default: the knowledge base's own settings)")
	flag.Var(&targets, "target", "system to evaluate (repeatable): name=,kind=external|dify,url=,key=,knowledge=[,retrieval=file.json]")
	flag.Parse()
	if *setPath == "" || len(targets) == 0 {
		flag.Usage()
		os.Exit(2)
	}
	qs, err := loadSet(*setPath)
	if err != nil {
		fatal(err)
	}
	var model json.RawMessage
	if *difyModel != "" {
		model = json.RawMessage(*difyModel)
		if !json.Valid(model) {
			fatal(fmt.Errorf("-dify-retrieval-model is not valid JSON"))
		}
	}

	var sums []summary
	for _, tg := range targets {
		fmt.Fprintf(os.Stderr, "evaluating %s (%d questions)...\n", tg.Name, len(qs))
		m := model
		if tg.Retrieval != nil {
			m = tg.Retrieval
		}
		sums = append(sums, evaluate(tg, qs, *k, *budget, *conc, m))
	}
	printTable(os.Stdout, sums)
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(sums, "", "  ")
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			fatal(err)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *jsonOut)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "rageval:", err)
	os.Exit(1)
}

func loadSet(p string) ([]question, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var qs []question
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for line := 1; sc.Scan(); line++ {
		text := strings.TrimSpace(sc.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		var q question
		if err := json.Unmarshal([]byte(text), &q); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", p, line, err)
		}
		if q.Query == "" || (len(q.Expect) == 0 && len(q.ExpectDocs) == 0) {
			return nil, fmt.Errorf("%s:%d: need query and expect or expect_docs", p, line)
		}
		if q.ID == "" {
			q.ID = fmt.Sprintf("line%d", line)
		}
		qs = append(qs, q)
	}
	return qs, sc.Err()
}

func evaluate(tg target, qs []question, k, budget, conc int, difyModel json.RawMessage) summary {
	outs := make([]outcome, len(qs))
	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	client := &http.Client{Timeout: 60 * time.Second}
	for i, q := range qs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			start := time.Now()
			recs, err := fetch(client, tg, q.Query, k, difyModel)
			o := outcome{QuestionID: q.ID, Query: q.Query, Latency: float64(time.Since(start).Microseconds()) / 1000}
			if err != nil {
				o.Error = err.Error()
			} else {
				score(&o, q, recs, k, budget)
			}
			outs[i] = o
		}()
	}
	wg.Wait()

	s := summary{Target: tg.Name, K: k, N: len(qs), Budget: budget, Outcomes: outs}
	var lat []float64
	for _, o := range outs {
		if o.Error != "" {
			s.Errors++
		}
		// Errors count as misses: a system that fails to answer is not excused.
		s.Recall += o.Recall
		s.Hit += o.Hit
		s.MRR += o.RR
		s.NDCG += o.NDCG
		s.RecallB += o.RecallBudget
		s.AvgChars += float64(o.Chars)
		lat = append(lat, o.Latency)
	}
	n := float64(len(outs))
	s.Recall, s.Hit, s.MRR, s.NDCG = s.Recall/n, s.Hit/n, s.MRR/n, s.NDCG/n
	s.RecallB, s.AvgChars = s.RecallB/n, s.AvgChars/n
	sort.Float64s(lat)
	s.P50, s.P95 = percentile(lat, 0.50), percentile(lat, 0.95)
	return s
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	return sorted[min(len(sorted)-1, int(math.Ceil(p*float64(len(sorted))))-1)]
}

func fetch(c *http.Client, tg target, query string, k int, difyModel json.RawMessage) ([]record, error) {
	var url string
	var body any
	if tg.Kind == "external" {
		url = tg.URL + "/retrieval"
		body = map[string]any{"knowledge_id": tg.Knowledge, "query": query,
			"retrieval_setting": map[string]any{"top_k": k, "score_threshold": 0.0}}
	} else {
		url = tg.URL + "/datasets/" + tg.Knowledge + "/retrieve"
		req := map[string]any{"query": query}
		if difyModel != nil {
			req["retrieval_model"] = difyModel
		}
		body = req
	}
	raw, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if tg.Key != "" {
		req.Header.Set("Authorization", "Bearer "+tg.Key)
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %.200s", resp.StatusCode, data)
	}
	if tg.Kind == "external" {
		var out struct {
			Records []record `json:"records"`
		}
		err = json.Unmarshal(data, &out)
		return out.Records, err
	}
	var out struct {
		Records []struct {
			Score   float64 `json:"score"`
			Segment struct {
				Content  string `json:"content"`
				Document struct {
					Name string `json:"name"`
				} `json:"document"`
			} `json:"segment"`
		} `json:"records"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	recs := make([]record, len(out.Records))
	for i, r := range out.Records {
		recs[i] = record{Content: r.Segment.Content, Title: r.Segment.Document.Name, Score: r.Score}
	}
	return recs, nil
}

// normalize drops whitespace and Markdown formatting characters and folds
// case, so line wrapping and how each system's extractor renders tables or
// emphasis do not decide relevance.
func normalize(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) || strings.ContainsRune("|*`#", r) {
			return -1
		}
		return unicode.ToLower(r)
	}, s)
}

func score(o *outcome, q question, recs []record, k, budget int) {
	if len(recs) > k {
		recs = recs[:k]
	}
	expect := make([]string, len(q.Expect))
	for i, e := range q.Expect {
		expect[i] = normalize(e)
	}
	docs := map[string]bool{}
	for _, d := range q.ExpectDocs {
		docs[normalize(path.Base(d))] = true
	}

	found := map[string]bool{} // expected phrases or documents seen in the top k
	var dcg float64
	for i, r := range recs {
		content, title := normalize(r.Content), normalize(path.Base(r.Title))
		rel := false
		if len(expect) > 0 {
			for _, e := range expect {
				if strings.Contains(content, e) {
					rel, found[e] = true, true
				}
			}
		} else if docs[title] {
			rel, found[title] = true, true
		}
		o.Relevant = append(o.Relevant, rel)
		o.Titles = append(o.Titles, r.Title)
		if rel {
			if o.RR == 0 {
				o.RR = 1 / float64(i+1)
			}
			dcg += 1 / math.Log2(float64(i+2))
		}
	}
	want := len(expect)
	if want == 0 {
		want = len(docs)
	}
	o.Recall = float64(len(found)) / float64(want)

	// recall@budget: the first `budget` characters, in rank order.
	var window strings.Builder
	left := budget
	for _, r := range recs {
		o.Chars += len([]rune(r.Content))
		if left <= 0 {
			continue
		}
		rs := []rune(r.Content)
		if len(rs) > left {
			rs = rs[:left]
		}
		window.WriteString(string(rs))
		window.WriteString("\n")
		left -= len(rs)
	}
	if len(expect) > 0 {
		text, inBudget := normalize(window.String()), 0
		for _, e := range expect {
			if strings.Contains(text, e) {
				inBudget++
			}
		}
		o.RecallBudget = float64(inBudget) / float64(len(expect))
	} else {
		o.RecallBudget = o.Recall // document-level questions: length does not matter
	}
	if len(found) > 0 {
		o.Hit = 1
	}
	var idcg float64
	for i := 0; i < min(want, k); i++ {
		idcg += 1 / math.Log2(float64(i+2))
	}
	if idcg > 0 {
		o.NDCG = math.Min(1, dcg/idcg)
	}
}

func printTable(w io.Writer, sums []summary) {
	if len(sums) == 0 {
		return
	}
	k := sums[0].K
	fmt.Fprintf(w, "\n| system | questions | errors | Recall@%d | Recall@%d chars | Hit@%d | MRR | nDCG@%d | avg chars | P50 ms | P95 ms |\n",
		k, sums[0].Budget, k, k)
	fmt.Fprintln(w, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
	for _, s := range sums {
		fmt.Fprintf(w, "| %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f | %.0f | %.0f |\n",
			s.Target, s.N, s.Errors, s.Recall, s.RecallB, s.Hit, s.MRR, s.NDCG, s.AvgChars, s.P50, s.P95)
	}
	if len(sums) == 2 {
		a, b := sums[0], sums[1]
		var onlyA, onlyB int
		for i := range a.Outcomes {
			if a.Outcomes[i].Hit > 0 && b.Outcomes[i].Hit == 0 {
				onlyA++
			}
			if b.Outcomes[i].Hit > 0 && a.Outcomes[i].Hit == 0 {
				onlyB++
			}
		}
		fmt.Fprintf(w, "\nquestions only %s answers: %d · only %s answers: %d (details per question in -json)\n",
			a.Target, onlyA, b.Target, onlyB)
	}
}
