package lexical

import (
	"slices"
	"testing"
	"time"
)

func TestTerms(t *testing.T) {
	start := time.Now()
	a, err := New()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("dictionary loaded in %s", time.Since(start).Round(time.Millisecond))

	terms := a.Terms("员工每年享有带薪年假，VPN 客户端在内网门户下载。")
	t.Logf("terms: %q", terms)
	for _, want := range []string{"带薪", "vpn", "客户端", "门户", "下载", "2:年假", "2:内网"} {
		if !slices.Contains(terms, want) {
			t.Errorf("missing term %q", want)
		}
	}
	for _, unwanted := range []string{"，", "。", " ", "在", "年", "假"} {
		if slices.Contains(terms, unwanted) {
			t.Errorf("punctuation/stop word %q kept", unwanted)
		}
	}
}

func TestDocumentWeights(t *testing.T) {
	a, err := New()
	if err != nil {
		t.Fatal(err)
	}
	doc := a.Document("年假，年假，年假。报销")
	weight := func(term string) float32 {
		i := slices.Index(doc.Indices, TermID(term))
		if i < 0 {
			t.Fatalf("term %q missing", term)
		}
		return doc.Values[i]
	}
	if weight("2:年假") <= weight("报销") {
		t.Errorf("repeated term should weigh more")
	}
	if weight("2:年假") >= float32(k1+1) {
		t.Errorf("BM25 tf component must saturate below k1+1, got %v", weight("2:年假"))
	}
	if !slices.IsSorted(doc.Indices) {
		t.Errorf("indices must be sorted")
	}
	q := a.Query("年假，年假")
	if len(q.Indices) != 1 || q.Values[0] != 1 {
		t.Errorf("query = %+v", q)
	}
}
