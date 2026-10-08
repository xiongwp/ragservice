package main

import (
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestScorePhrases(t *testing.T) {
	q := question{Query: "q", Expect: []string{"年假为十天", "Annual Leave"}}
	recs := []record{
		{Content: "无关内容"},
		{Content: "满十年的，年假为\n十天。"}, // line break inside the phrase still counts
		{Content: "see annual   leave policy"},
		{Content: "年假为十天"}, // beyond k
	}
	var o outcome
	score(&o, q, recs, 3, 1000)
	if !near(o.Recall, 1) || o.Hit != 1 || !near(o.RR, 0.5) {
		t.Errorf("recall=%v hit=%v rr=%v", o.Recall, o.Hit, o.RR)
	}
	// relevant at ranks 2 and 3; ideal is ranks 1 and 2.
	want := (1/math.Log2(3) + 1/math.Log2(4)) / (1 + 1/math.Log2(3))
	if !near(o.NDCG, want) {
		t.Errorf("ndcg = %v, want %v", o.NDCG, want)
	}
}

func TestScoreDocs(t *testing.T) {
	q := question{Query: "q", ExpectDocs: []string{"docs/IT 指南.pdf"}}
	var o outcome
	score(&o, q, []record{{Title: "员工手册.md"}, {Title: "IT 指南.pdf"}}, 5, 1000)
	if o.Hit != 1 || !near(o.RR, 0.5) || !near(o.Recall, 1) {
		t.Errorf("%+v", o)
	}
	var miss outcome
	score(&miss, q, []record{{Title: "other.pdf"}}, 5, 1000)
	if miss.Hit != 0 || miss.RR != 0 || miss.NDCG != 0 {
		t.Errorf("%+v", miss)
	}
}

func TestFetchBothFormats(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			http.Error(w, "no", http.StatusForbidden)
			return
		}
		switch r.URL.Path {
		case "/retrieval":
			w.Write([]byte(`{"records":[{"content":"A","title":"a.md","score":0.9}]}`))
		case "/v1/datasets/ds/retrieve":
			w.Write([]byte(`{"query":{"content":"q"},"records":[{"score":0.8,"segment":{"content":"B","document":{"name":"b.md"}}}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := srv.Client()

	recs, err := fetch(c, target{Kind: "external", URL: srv.URL, Key: "k", Knowledge: "ds"}, "q", 5, nil)
	if err != nil || len(recs) != 1 || recs[0].Content != "A" || recs[0].Title != "a.md" {
		t.Errorf("external: %v %v", recs, err)
	}
	recs, err = fetch(c, target{Kind: "dify", URL: srv.URL + "/v1", Key: "k", Knowledge: "ds"}, "q", 5, nil)
	if err != nil || len(recs) != 1 || recs[0].Content != "B" || recs[0].Title != "b.md" {
		t.Errorf("dify: %v %v", recs, err)
	}
	if _, err := fetch(c, target{Kind: "external", URL: srv.URL, Key: "bad", Knowledge: "ds"}, "q", 5, nil); err == nil {
		t.Error("want error on 403")
	}
}

func TestTargetFlag(t *testing.T) {
	var ts targetFlags
	if err := ts.Set("name=ragd,kind=external,url=http://x/,key=k,knowledge=1"); err != nil {
		t.Fatal(err)
	}
	if ts[0].URL != "http://x" || ts[0].Name != "ragd" {
		t.Errorf("%+v", ts[0])
	}
	for _, bad := range []string{"kind=foo,url=x,knowledge=1", "kind=dify,url=x", "kind=dify,url=x,knowledge=1,nope"} {
		if err := ts.Set(bad); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
	}
}

func TestPercentile(t *testing.T) {
	xs := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if percentile(xs, 0.5) != 5 || percentile(xs, 0.95) != 10 || percentile(nil, 0.5) != 0 {
		t.Error("percentile")
	}
}

func TestRecallAtBudget(t *testing.T) {
	q := question{Query: "q", Expect: []string{"答案在这里"}}
	long := strings.Repeat("填充", 600) + "答案在这里" // the phrase sits after 1200 chars
	var o outcome
	score(&o, q, []record{{Content: long}}, 5, 1000)
	if o.Recall != 1 || o.RecallBudget != 0 {
		t.Errorf("recall=%v budget=%v: a phrase beyond the budget must not count", o.Recall, o.RecallBudget)
	}
	var o2 outcome
	score(&o2, q, []record{{Content: "短"}, {Content: "答案在这里"}}, 5, 1000)
	if o2.RecallBudget != 1 || o2.Chars != 6 {
		t.Errorf("budget=%v chars=%d", o2.RecallBudget, o2.Chars)
	}
}

func TestWindows(t *testing.T) {
	text := "# 标题\n\n" + strings.Repeat("这是一段足够长的正文内容。", 15) + "\n\n短\n\n" + strings.Repeat("另一段正文。", 30)
	ws := windows(text)
	if len(ws) != 2 {
		t.Fatalf("got %d windows: %q", len(ws), ws)
	}
	if !strings.HasPrefix(ws[0], "# 标题") {
		t.Errorf("heading should stay with its paragraph: %q", ws[0][:20])
	}
}
