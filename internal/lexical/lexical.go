// Package lexical turns text into BM25-weighted sparse vectors for the
// keyword half of hybrid search.
//
// Chinese gets two kinds of terms: dictionary words from gse (a Go port of
// jieba, search mode, so long compounds also yield their parts) for
// precision, and overlapping character bigrams for recall. Bigrams matter
// because company documents are full of words no general dictionary has:
// gse splits 年假 into 年 / 假 and 内网 into 在内 / 网, but the bigrams 年假
// and 内网 still match. Single-character words are dropped (the bigrams cover
// them); IDF keeps the noisy bigrams from dominating.
//
// Each term is hashed to a sparse-vector index. Documents carry the BM25
// term-frequency component; Qdrant multiplies in the IDF (the collection's
// sparse vector uses Modifier=IDF), so the score of a query against a chunk
// is classic BM25 without us tracking corpus statistics.
package lexical

import (
	"hash/fnv"
	"sort"
	"strings"
	"unicode"

	"github.com/go-ego/gse"

	"rag/internal/vector"
)

const (
	k1 = 1.2
	b  = 0.75
	// avgDocLen approximates the average child chunk length in terms. Exact
	// corpus statistics would need a global pass; a constant keeps indexing
	// independent per document and is close enough for length normalisation.
	avgDocLen = 100.0
)

type Analyzer struct {
	seg gse.Segmenter
}

// New loads the embedded simplified-Chinese dictionary and stop words
// (about a second, once per process).
func New() (*Analyzer, error) {
	a := &Analyzer{}
	a.seg.SkipLog = true
	if err := a.seg.LoadDictEmbed("zh_s"); err != nil {
		return nil, err
	}
	if err := a.seg.LoadStopEmbed(); err != nil {
		return nil, err
	}
	return a, nil
}

// Terms returns the normalised terms of text, with repeats: words first,
// then bigrams (prefixed "2:" so a bigram and an identical word are
// distinct terms with their own IDF).
func (a *Analyzer) Terms(text string) []string {
	var out []string
	for _, tok := range a.seg.CutSearch(text, true) {
		t := strings.ToLower(strings.TrimSpace(tok))
		if t == "" || !hasWordRune(t) || a.seg.IsStop(t) {
			continue
		}
		if r := []rune(t); len(r) == 1 && isHan(r[0]) {
			continue
		}
		out = append(out, t)
	}
	var run []rune
	flush := func() {
		for i := 0; i+1 < len(run); i++ {
			out = append(out, "2:"+string(run[i:i+2]))
		}
		run = run[:0]
	}
	for _, r := range text {
		if isHan(r) {
			run = append(run, r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

func isHan(r rune) bool { return unicode.Is(unicode.Han, r) }

func hasWordRune(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return true
		}
	}
	return false
}

// Document returns the BM25 term-frequency weights of a chunk.
func (a *Analyzer) Document(text string) *vector.Sparse {
	terms := a.Terms(text)
	tf := map[string]float64{}
	for _, t := range terms {
		tf[t]++
	}
	norm := k1 * (1 - b + b*float64(len(terms))/avgDocLen)
	weights := make(map[uint32]float32, len(tf))
	for t, f := range tf {
		weights[TermID(t)] += float32(f * (k1 + 1) / (f + norm))
	}
	return build(weights)
}

// Query returns the query's distinct terms, each with weight 1.
func (a *Analyzer) Query(text string) *vector.Sparse {
	weights := map[uint32]float32{}
	for _, t := range a.Terms(text) {
		weights[TermID(t)] = 1
	}
	return build(weights)
}

func TermID(term string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(term))
	return h.Sum32()
}

// build sorts by index; Qdrant requires unique indices, which the map
// guarantees even when two terms hash alike (their weights are summed).
func build(weights map[uint32]float32) *vector.Sparse {
	s := &vector.Sparse{Indices: make([]uint32, 0, len(weights)), Values: make([]float32, 0, len(weights))}
	for i := range weights {
		s.Indices = append(s.Indices, i)
	}
	sort.Slice(s.Indices, func(x, y int) bool { return s.Indices[x] < s.Indices[y] })
	for _, i := range s.Indices {
		s.Values = append(s.Values, weights[i])
	}
	return s
}
