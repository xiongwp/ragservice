package parse

import (
	"os"
	"strings"
	"testing"
	"time"

	"golang.org/x/text/encoding/simplifiedchinese"
)

func TestFormatOf(t *testing.T) {
	for name, want := range map[string]string{"a.MD": "md", "b.htm": "html", "c.pdf": "pdf", "d.exe": "", "noext": ""} {
		if got := FormatOf(name); got != want {
			t.Errorf("FormatOf(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestParsePlainFormats(t *testing.T) {
	p := New("", time.Minute, 1)
	gbk, _ := simplifiedchinese.GB18030.NewEncoder().String("中文文本")
	for _, tc := range []struct{ format, in, want string }{
		{"txt", "\xEF\xBB\xBFhello", "hello"},
		{"txt", gbk, "中文文本"},
		{"html", "<h2>标题</h2><p>正文</p>", "## 标题\n\n正文"},
	} {
		got, err := p.Parse(t.Context(), "x", tc.format, []byte(tc.in))
		if err != nil || strings.TrimSpace(got) != tc.want {
			t.Errorf("%s %q: got %q, %v; want %q", tc.format, tc.in, got, err, tc.want)
		}
	}
	if _, err := p.Parse(t.Context(), "x.pdf", "pdf", []byte("%PDF")); err == nil {
		t.Error("pdf without docling should fail")
	}
}

// Runs only with a docling-serve instance: RAG_TEST_DOCLING_URL=http://localhost:5001
func TestDocling(t *testing.T) {
	url := os.Getenv("RAG_TEST_DOCLING_URL")
	if url == "" {
		t.Skip("RAG_TEST_DOCLING_URL not set")
	}
	data, err := os.ReadFile("testdata/sample.pdf")
	if err != nil {
		t.Fatal(err)
	}
	md, err := New(url, 30*time.Minute, 1).Parse(t.Context(), "sample.pdf", "pdf", data)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("markdown:\n%s", md)
	for _, want := range []string{"Refund Policy", "30 days", "Annual", "|"} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown is missing %q", want)
		}
	}
}
