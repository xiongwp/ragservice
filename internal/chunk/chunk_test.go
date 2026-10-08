package chunk

import (
	"strings"
	"testing"
)

func tokenizer(t *testing.T) Tokenizer {
	t.Helper()
	tok, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestSplitRespectsLimitsAndHeadings(t *testing.T) {
	tok := tokenizer(t)
	var b strings.Builder
	b.WriteString("# 员工手册\n\n")
	b.WriteString("## 请假制度\n\n")
	for i := 0; i < 40; i++ {
		b.WriteString("员工每年享有带薪年假，年假天数根据工龄计算，工龄满一年不满十年的为五天。")
	}
	b.WriteString("\n\n## 报销流程\n\n")
	b.WriteString("1. 填写报销单\n2. 部门经理审批\n3. 财务复核后打款\n\n")
	b.WriteString("| 类型 | 上限 |\n|---|---|\n")
	for i := 0; i < 60; i++ {
		b.WriteString("| 差旅住宿 | 每晚 500 元 |\n")
	}

	opts := Options{ParentTokens: 300, ChildTokens: 100}
	parents := Split(Clean(b.String()), opts, tok)
	if len(parents) < 3 {
		t.Fatalf("expected several parents, got %d", len(parents))
	}
	for i, p := range parents {
		if p.Tokens > opts.ParentTokens+opts.ParentTokens/10 {
			t.Errorf("parent %d has %d tokens, limit %d", i, p.Tokens, opts.ParentTokens)
		}
		if len(p.Children) == 0 {
			t.Errorf("parent %d has no children", i)
		}
		for j, c := range p.Children {
			if c.Tokens > opts.ChildTokens+opts.ChildTokens/10 {
				t.Errorf("parent %d child %d has %d tokens, limit %d", i, j, c.Tokens, opts.ChildTokens)
			}
			if strings.TrimSpace(c.Content) == "" {
				t.Errorf("parent %d child %d is empty", i, j)
			}
		}
	}

	// The reimbursement section must not be merged into the leave section's parent.
	for _, p := range parents {
		if strings.Contains(p.Content, "带薪年假") && strings.Contains(p.Content, "报销") {
			t.Errorf("parent mixes two large sections:\n%s", p.Content)
		}
	}

	// Every table piece repeats the header row.
	for _, p := range parents {
		for _, c := range p.Children {
			if strings.Contains(c.Content, "差旅住宿") && !strings.HasPrefix(c.Content, "| 类型 | 上限 |") &&
				!strings.Contains(c.Content, "| 类型 | 上限 |") {
				t.Errorf("table child lost its header:\n%s", c.Content)
			}
		}
	}

	// Headings are tracked as paths.
	var sawPath bool
	for _, p := range parents {
		for _, c := range p.Children {
			if c.Heading == "员工手册 > 报销流程" {
				sawPath = true
			}
		}
	}
	if !sawPath {
		t.Errorf("no child carries heading path 员工手册 > 报销流程")
	}
}

func TestSplitKeepsAllText(t *testing.T) {
	tok := tokenizer(t)
	text := "# A\n\nfirst paragraph.\n\n## B\n\n- one\n- two\n\n```go\nfunc main() {}\n```\n\nlast words."
	parents := Split(text, Options{ParentTokens: 800, ChildTokens: 200}, tok)
	if len(parents) != 1 {
		t.Fatalf("small doc should be one parent, got %d", len(parents))
	}
	for _, want := range []string{"# A", "first paragraph.", "## B", "- two", "func main() {}", "last words."} {
		if !strings.Contains(parents[0].Content, want) {
			t.Errorf("parent lost %q", want)
		}
	}
	if len(parents[0].Children) != 1 || parents[0].Children[0].Content != parents[0].Content {
		t.Errorf("a parent under the child limit should be its own single child")
	}
}

func TestBareHeadingJoinsNextSection(t *testing.T) {
	tok := tokenizer(t)
	body := strings.Repeat("Some body text about the topic. ", 30)
	text := "# Title\n\n## Intro\n\n" + body + "\n\n## Next\n\n" + body
	parents := Split(text, Options{ParentTokens: 2000, ChildTokens: 120}, tok)
	for _, p := range parents {
		for _, c := range p.Children {
			if strings.TrimSpace(c.Content) == "# Title" || strings.TrimSpace(c.Content) == "# Title\n\n## Intro" {
				t.Errorf("heading-only child: %q", c.Content)
			}
		}
	}
}

func TestHardSplitLongLine(t *testing.T) {
	tok := tokenizer(t)
	line := strings.Repeat("无标点的超长文本", 400)
	parents := Split(line, Options{ParentTokens: 200, ChildTokens: 50}, tok)
	var joined strings.Builder
	for _, p := range parents {
		joined.WriteString(p.Content)
		if p.Tokens > 200 {
			t.Errorf("parent over limit: %d", p.Tokens)
		}
	}
	if joined.String() != line {
		t.Errorf("hard split lost or corrupted text")
	}
}

func TestSentences(t *testing.T) {
	got := sentences("你好。“是吗？”他说。Yes. e.g.x ok")
	want := []string{"你好。", "“是吗？”", "他说。", "Yes. ", "e.g.x ok"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestClean(t *testing.T) {
	in := "a\r\nb\x00c\u200b\n\n\n\n d \t\n"
	if got := Clean(in); got != "a\nbc\n\n d" {
		t.Errorf("Clean = %q", got)
	}
}

func TestResolve(t *testing.T) {
	def := Options{ParentTokens: 800, ChildTokens: 200}
	for _, tc := range []struct {
		name    string
		in      Options
		want    Options
		wantErr string
	}{
		{"both defaulted", Options{}, Options{800, 200}, ""},
		{"parent set", Options{ParentTokens: 1200}, Options{1200, 200}, ""},
		{"child set", Options{ChildTokens: 100}, Options{800, 100}, ""},
		{"child too small", Options{ChildTokens: 10}, Options{}, "child_tokens (10) must be >= 32"},
		{"negative child", Options{ChildTokens: -5}, Options{}, "child_tokens (-5)"},
		// The case that used to slip through at index time: an explicit
		// parent smaller than a default child raised after the dataset was made.
		{"parent below default child", Options{ParentTokens: 150}, Options{}, "parent_tokens (150) must be >= child_tokens (200)"},
		{"negative parent", Options{ParentTokens: -1}, Options{}, "parent_tokens (-1)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Resolve(tc.in, def)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("Resolve(%+v) error = %v, want %q", tc.in, err, tc.wantErr)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("Resolve(%+v) = %+v, %v; want %+v", tc.in, got, err, tc.want)
			}
		})
	}
}
