// Package chunk splits Markdown into parent/child chunks.
//
// Splitting follows document structure first (headings, tables, lists, code
// fences, paragraphs) and only then packs blocks by token count, so a table
// row or a sentence is never cut in half unless it alone exceeds the limit.
// Children (small, precise) are what gets embedded and matched; parents (a
// section-sized window) are what gets returned to the LLM.
package chunk

import (
	"fmt"
	"regexp"
	"strings"
)

type Options struct {
	ParentTokens int
	ChildTokens  int
}

// MinChildTokens is the smallest usable child: below it a child is a few
// words, too little for an embedding to place.
const MinChildTokens = 32

// Resolve takes the sizes left at zero in o from def, then checks that the
// result can be split with: children of at least MinChildTokens that fit in
// their parents.
func Resolve(o, def Options) (Options, error) {
	if o.ParentTokens == 0 {
		o.ParentTokens = def.ParentTokens
	}
	if o.ChildTokens == 0 {
		o.ChildTokens = def.ChildTokens
	}
	if o.ChildTokens < MinChildTokens {
		return o, fmt.Errorf("child_tokens (%d) must be >= %d", o.ChildTokens, MinChildTokens)
	}
	if o.ParentTokens < o.ChildTokens {
		return o, fmt.Errorf("parent_tokens (%d) must be >= child_tokens (%d)", o.ParentTokens, o.ChildTokens)
	}
	return o, nil
}

type Parent struct {
	Heading  string // "H1 > H2" path in effect where the parent starts
	Content  string
	Tokens   int
	Children []Child
}

type Child struct {
	Heading string
	Content string
	Tokens  int
}

type kind int

const (
	kindPara kind = iota
	kindHeading
	kindList
	kindTable
	kindCode
)

type block struct {
	kind    kind
	text    string
	heading string
	tokens  int
}

// Split chunks cleaned Markdown. Empty input yields no parents.
func Split(text string, opts Options, tok Tokenizer) []Parent {
	s := splitter{opts: opts, tok: tok}
	blocks := parseBlocks(text)
	for i := range blocks {
		blocks[i].tokens = tok.Count(blocks[i].text)
	}
	return s.parents(blocks)
}

type splitter struct {
	opts Options
	tok  Tokenizer
}

func (s splitter) parents(blocks []block) []Parent {
	var out []Parent
	var cur []block
	curTokens := 0
	// A new heading starts a new parent once the current one has some body;
	// tiny sections are merged forward instead of becoming tiny parents.
	minFlush := s.opts.ParentTokens / 4

	flush := func(carryHeadings bool) {
		var carry []block
		if carryHeadings {
			// Never leave a heading dangling at the end of a parent.
			for len(cur) > 0 && cur[len(cur)-1].kind == kindHeading {
				carry = append([]block{cur[len(cur)-1]}, carry...)
				cur = cur[:len(cur)-1]
			}
		}
		if len(cur) > 0 {
			out = append(out, s.makeParent(cur))
		}
		cur, curTokens = carry, 0
		for _, b := range carry {
			curTokens += b.tokens
		}
	}

	for _, b := range blocks {
		if b.kind == kindHeading && curTokens >= minFlush {
			flush(false)
		}
		if b.tokens > s.opts.ParentTokens {
			flush(true)
			pieces := s.splitBlock(b, s.opts.ParentTokens)
			for i, p := range pieces {
				if i == 0 && len(cur) > 0 {
					// carried headings belong in front of the first piece
					cur = append(cur, p)
					flush(false)
					continue
				}
				out = append(out, s.makeParent([]block{p}))
			}
			continue
		}
		if curTokens+b.tokens > s.opts.ParentTokens {
			flush(true)
		}
		cur = append(cur, b)
		curTokens += b.tokens
	}
	flush(false)
	return out
}

func (s splitter) makeParent(blocks []block) Parent {
	texts := make([]string, len(blocks))
	for i, b := range blocks {
		texts[i] = b.text
	}
	p := Parent{Heading: blocks[0].heading, Content: strings.Join(texts, "\n\n")}
	p.Tokens = s.tok.Count(p.Content)
	if p.Tokens <= s.opts.ChildTokens {
		p.Children = []Child{{Heading: p.Heading, Content: p.Content, Tokens: p.Tokens}}
		return p
	}
	// Children never straddle a heading: each run of blocks under the same
	// heading path is packed on its own (a heading block carries its own
	// path, so it stays with the body that follows it).
	for start := 0; start < len(blocks); {
		end := start + 1
		for end < len(blocks) && blocks[end].heading == blocks[start].heading {
			end++
		}
		heading := blocks[start].heading
		// A run of bare headings ("# Doc" right before "## Intro") joins the
		// run below it rather than becoming a child with no body.
		for end < len(blocks) && onlyHeadings(blocks[start:end]) {
			heading = blocks[end].heading
			for end++; end < len(blocks) && blocks[end].heading == heading; end++ {
			}
		}
		var units []string
		for _, b := range blocks[start:end] {
			if b.tokens <= s.opts.ChildTokens {
				units = append(units, b.text)
				continue
			}
			for _, piece := range s.splitBlock(b, s.opts.ChildTokens) {
				units = append(units, piece.text)
			}
		}
		for _, c := range s.pack(units, "\n\n", s.opts.ChildTokens, "") {
			p.Children = append(p.Children, Child{Heading: heading, Content: c, Tokens: s.tok.Count(c)})
		}
		start = end
	}
	return p
}

func onlyHeadings(blocks []block) bool {
	for _, b := range blocks {
		if b.kind != kindHeading {
			return false
		}
	}
	return true
}

// splitBlock breaks one oversized block along its natural units.
func (s splitter) splitBlock(b block, max int) []block {
	var pieces []string
	switch b.kind {
	case kindTable:
		header, rows := tableHeader(b.text)
		pieces = s.pack(rows, "\n", max-s.tok.Count(header), header)
	case kindCode:
		lines := strings.Split(b.text, "\n")
		open, close := lines[0], ""
		if len(lines) > 1 && isFence(lines[len(lines)-1]) {
			close = lines[len(lines)-1]
			lines = lines[1 : len(lines)-1]
		} else {
			lines = lines[1:]
		}
		for _, p := range s.pack(lines, "\n", max-s.tok.Count(open+close)-2, "") {
			pieces = append(pieces, open+"\n"+p+"\n"+close)
		}
	case kindList:
		pieces = s.pack(listItems(b.text), "\n", max, "")
	default:
		pieces = s.pack(sentences(b.text), "", max, "")
	}
	if len(pieces) == 0 { // e.g. a fence with no body lines
		pieces = s.hardSplit(b.text, max)
	}
	out := make([]block, 0, len(pieces))
	for _, p := range pieces {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		out = append(out, block{kind: b.kind, text: p, heading: b.heading, tokens: s.tok.Count(p)})
	}
	return out
}

// pack greedily joins units into pieces of at most max tokens, each prefixed
// with prefix (a table header). Token counts of joined units are summed rather
// than recounted, which can overshoot by a token or two at the seams.
func (s splitter) pack(units []string, sep string, max int, prefix string) []string {
	if max < 8 {
		max = 8
	}
	var out []string
	var buf strings.Builder
	bufTokens := 0
	emit := func() {
		if buf.Len() == 0 {
			return
		}
		if prefix != "" {
			out = append(out, prefix+"\n"+buf.String())
		} else {
			out = append(out, buf.String())
		}
		buf.Reset()
		bufTokens = 0
	}
	for _, u := range units {
		t := s.tok.Count(u)
		if t > max {
			emit()
			for _, h := range s.hardSplit(u, max) {
				buf.WriteString(h)
				emit()
			}
			continue
		}
		if bufTokens > 0 && bufTokens+t > max {
			emit()
		}
		if buf.Len() > 0 {
			buf.WriteString(sep)
		}
		buf.WriteString(u)
		bufTokens += t
	}
	emit()
	return out
}

// hardSplit is the last resort for a single unit with no natural boundary
// (a 3,000-character line). It cuts on rune boundaries, never mid-character.
func (s splitter) hardSplit(text string, max int) []string {
	runes := []rune(text)
	var out []string
	for len(runes) > 0 {
		if s.tok.Count(string(runes)) <= max {
			out = append(out, string(runes))
			break
		}
		lo, hi := 1, len(runes)
		for lo < hi { // largest prefix that fits
			mid := (lo + hi + 1) / 2
			if s.tok.Count(string(runes[:mid])) <= max {
				lo = mid
			} else {
				hi = mid - 1
			}
		}
		out = append(out, string(runes[:lo]))
		runes = runes[lo:]
	}
	return out
}

var (
	headingRe  = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)
	listItemRe = regexp.MustCompile(`^\s*([-*+]|\d{1,3}[.)])\s+`)
	tableSepRe = regexp.MustCompile(`^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$`)
)

func isFence(line string) bool {
	t := strings.TrimSpace(line)
	return strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~")
}

func parseBlocks(text string) []block {
	lines := strings.Split(text, "\n")
	var blocks []block
	var headings [6]string
	path := ""
	var cur []string
	curKind := kindPara

	flush := func() {
		if len(cur) > 0 {
			if t := strings.TrimSpace(strings.Join(cur, "\n")); t != "" {
				blocks = append(blocks, block{kind: curKind, text: t, heading: path})
			}
		}
		cur, curKind = nil, kindPara
	}

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)
		switch {
		case isFence(line):
			flush()
			marker := trimmed[:3]
			code := []string{line}
			for i++; i < len(lines); i++ {
				code = append(code, lines[i])
				if strings.HasPrefix(strings.TrimSpace(lines[i]), marker) {
					break
				}
			}
			cur, curKind = code, kindCode
			flush()
		case headingRe.MatchString(line):
			flush()
			m := headingRe.FindStringSubmatch(line)
			level := len(m[1])
			headings[level-1] = m[2]
			for j := level; j < len(headings); j++ {
				headings[j] = ""
			}
			var parts []string
			for _, h := range headings {
				if h != "" {
					parts = append(parts, h)
				}
			}
			path = strings.Join(parts, " > ")
			blocks = append(blocks, block{kind: kindHeading, text: trimmed, heading: path})
		case trimmed == "":
			flush()
		case strings.HasPrefix(trimmed, "|"):
			if curKind != kindTable {
				flush()
				curKind = kindTable
			}
			cur = append(cur, line)
		case listItemRe.MatchString(line):
			if curKind != kindList {
				flush()
				curKind = kindList
			}
			cur = append(cur, line)
		default:
			if curKind == kindTable {
				flush()
			}
			// a non-item line inside a list is a continuation of the item
			cur = append(cur, line)
		}
	}
	flush()
	return blocks
}

// tableHeader returns the header row (+ separator row, if present) to repeat
// on every piece of a split table, and the remaining body rows.
func tableHeader(table string) (string, []string) {
	rows := strings.Split(table, "\n")
	if len(rows) >= 2 && tableSepRe.MatchString(strings.TrimSpace(rows[1])) {
		return rows[0] + "\n" + rows[1], rows[2:]
	}
	return rows[0], rows[1:]
}

func listItems(list string) []string {
	var items []string
	for _, line := range strings.Split(list, "\n") {
		if listItemRe.MatchString(line) || len(items) == 0 {
			items = append(items, line)
		} else {
			items[len(items)-1] += "\n" + line
		}
	}
	return items
}

// sentences splits after CJK and Latin sentence terminators and at line
// breaks, keeping the terminator and any closing quote with its sentence.
// The pieces concatenate back to the original text.
func sentences(text string) []string {
	var out []string
	runes := []rune(text)
	start := 0
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		end := false
		switch r {
		case '。', '！', '？', '；', '!', '?', ';', '…', '\n':
			end = true
		case '.':
			end = i+1 == len(runes) || runes[i+1] == ' ' || runes[i+1] == '\n'
		}
		if !end {
			continue
		}
		for i+1 < len(runes) && strings.ContainsRune("”’\"')）】」』 ", runes[i+1]) {
			i++
		}
		out = append(out, string(runes[start:i+1]))
		start = i + 1
	}
	if start < len(runes) {
		out = append(out, string(runes[start:]))
	}
	return out
}
