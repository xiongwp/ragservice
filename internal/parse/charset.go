package parse

import (
	"bytes"
	"strings"
	"unicode/utf8"

	"github.com/saintfish/chardet"
	"golang.org/x/net/html/charset"
	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/htmlindex"
	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/encoding/unicode/utf32"
)

// minConfidence is the chardet score (0–100) below which a guess is not
// trusted. Measured on paragraph-length samples: correct guesses scored
// 57–100 (GB18030, Big5, Shift_JIS, EUC-JP, EUC-KR, windows-1251/1252,
// ISO-8859-7), while a one-sentence GBK text and windows-874 Thai (which
// chardet has no model for) scored 10 for a wrong or tied answer.
const minConfidence = 50

// sniffBytes is how much of a file chardet looks at; enough for its
// statistics, and bounded so huge files stay cheap.
const sniffBytes = 64 << 10

var boms = []struct {
	prefix []byte
	enc    encoding.Encoding
}{
	// UTF-32 first: its little-endian BOM starts with the UTF-16 one.
	{[]byte{0xFF, 0xFE, 0x00, 0x00}, utf32.UTF32(utf32.LittleEndian, utf32.ExpectBOM)},
	{[]byte{0x00, 0x00, 0xFE, 0xFF}, utf32.UTF32(utf32.BigEndian, utf32.ExpectBOM)},
	{[]byte{0xEF, 0xBB, 0xBF}, unicode.UTF8BOM},
	{[]byte{0xFF, 0xFE}, unicode.UTF16(unicode.LittleEndian, unicode.ExpectBOM)},
	{[]byte{0xFE, 0xFF}, unicode.UTF16(unicode.BigEndian, unicode.ExpectBOM)},
}

// decodeText turns a plain-text file of unknown encoding into UTF-8:
//  1. a byte-order mark decides (UTF-8/16/32);
//  2. valid UTF-8 is taken as is (true for most files written today);
//  3. otherwise chardet's statistical guess, if it is confident;
//  4. otherwise fallback, the deployment's most likely legacy encoding.
//
// It reports the encoding it used, for logs and tests.
func decodeText(data []byte, fallback encoding.Encoding, fallbackName string) (string, string) {
	for _, b := range boms {
		if bytes.HasPrefix(data, b.prefix) {
			if out, err := b.enc.NewDecoder().Bytes(data); err == nil {
				return string(out), bomName(b.prefix)
			}
		}
	}
	if utf8.Valid(data) {
		return string(data), "utf-8"
	}
	sample := data
	if len(sample) > sniffBytes {
		sample = trimPartial(sample[:sniffBytes])
	}
	if r, err := chardet.NewTextDetector().DetectBest(sample); err == nil && r.Confidence >= minConfidence {
		if enc, name := lookup(r.Charset); enc != nil {
			if out, err := enc.NewDecoder().Bytes(data); err == nil {
				return string(out), name
			}
		}
	}
	if out, err := fallback.NewDecoder().Bytes(data); err == nil {
		return string(out), fallbackName
	}
	return strings.ToValidUTF8(string(data), "�"), "invalid"
}

// decodeHTML honours what the page declares (BOM, <meta charset>), as a
// browser would, and only guesses when nothing is declared.
func decodeHTML(data []byte, fallback encoding.Encoding, fallbackName string) (string, string) {
	enc, name, certain := charset.DetermineEncoding(data, "text/html")
	// Without a declaration or valid UTF-8, DetermineEncoding answers
	// windows-1252 by default; that is a guess, not a finding, so use the
	// statistical detector instead.
	if !certain && name == "windows-1252" && !declares(data, "windows-1252", "iso-8859-1", "latin1", "us-ascii") {
		return decodeText(data, fallback, fallbackName)
	}
	out, err := enc.NewDecoder().Bytes(data)
	if err != nil {
		return decodeText(data, fallback, fallbackName)
	}
	return string(out), name
}

// lookup maps a charset label (chardet's IANA names, WHATWG labels) to an
// encoding, using the WHATWG table browsers use (ISO-8859-1 → windows-1252).
func lookup(label string) (encoding.Encoding, string) {
	l := strings.ToLower(strings.TrimSpace(label))
	if l == "gb-18030" { // chardet's spelling
		l = "gb18030"
	}
	enc, err := htmlindex.Get(l)
	if err != nil {
		return nil, ""
	}
	name, _ := htmlindex.Name(enc)
	return enc, name
}

// FallbackEncoding resolves RAG_TEXT_FALLBACK_ENCODING.
func FallbackEncoding(label string) (encoding.Encoding, string, bool) {
	enc, name := lookup(label)
	return enc, name, enc != nil
}

// declares reports whether the first KB names one of the given charsets,
// i.e. a windows-1252 answer came from the page and not from the default.
func declares(data []byte, labels ...string) bool {
	head := strings.ToLower(string(data[:min(len(data), 1024)]))
	for _, l := range labels {
		if strings.Contains(head, "charset="+l) || strings.Contains(head, `charset="`+l) {
			return true
		}
	}
	return false
}

func bomName(prefix []byte) string {
	switch len(prefix) {
	case 4:
		return "utf-32"
	case 3:
		return "utf-8"
	}
	return "utf-16"
}

// trimPartial drops a multi-byte character cut in half by the sample limit.
func trimPartial(b []byte) []byte {
	for i := len(b) - 1; i >= 0 && i > len(b)-4; i-- {
		if b[i] < 0x80 {
			break
		}
		if b[i] >= 0xC0 {
			return b[:i]
		}
	}
	return b
}
