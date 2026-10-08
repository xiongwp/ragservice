package rewrite

import (
	"slices"
	"testing"
)

func TestParaphrases(t *testing.T) {
	out := "1. 年休假天数\n2）带薪年假怎么算\n- 年假有几天\n(4) 年休假天数\n\n• 第五个\n"
	got := paraphrases(out, "年假有几天", 3)
	want := []string{"年休假天数", "带薪年假怎么算", "第五个"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q (numbering stripped, original and repeats dropped)", got, want)
	}
}

func TestFirstLine(t *testing.T) {
	if got := firstLine("\n  “企业版多少钱”  \n解释……"); got != "企业版多少钱" {
		t.Errorf("got %q", got)
	}
}

func TestLRU(t *testing.T) {
	c := newLRU(2)
	c.put("a", Expansion{Query: "a"})
	c.put("b", Expansion{Query: "b"})
	c.get("a") // a is now most recent
	c.put("c", Expansion{Query: "c"})
	if _, ok := c.get("b"); ok {
		t.Error("b should have been evicted")
	}
	if e, ok := c.get("a"); !ok || e.Query != "a" {
		t.Error("a should still be cached")
	}
}
