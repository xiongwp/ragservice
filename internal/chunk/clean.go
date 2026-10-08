package chunk

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

var manyBlankLines = regexp.MustCompile(`\n{3,}`)

// Clean normalises extracted text without changing what a reader sees:
// NFC (not NFKC, which would turn full-width CJK punctuation into ASCII),
// LF line endings, no control or zero-width characters, no trailing spaces,
// and at most one blank line between blocks.
func Clean(s string) string {
	s = norm.NFC.String(s)
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.Map(func(r rune) rune {
		switch {
		case r == '\n' || r == '\t':
			return r
		case r == '\u00a0' || r == '\u3000':
			return ' '
		case r == '\u200b' || r == '\u200c' || r == '\u200d' || r == '\ufeff' || r == '\u00ad':
			return -1
		case unicode.IsControl(r):
			return -1
		}
		return r
	}, s)
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimRight(l, " \t")
	}
	s = strings.Join(lines, "\n")
	s = manyBlankLines.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}
