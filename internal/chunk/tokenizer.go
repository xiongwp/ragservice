package chunk

import (
	"fmt"

	"github.com/pkoukk/tiktoken-go"
	tiktokenloader "github.com/pkoukk/tiktoken-go-loader"
)

// Tokenizer counts tokens. cl100k_base is not the embedding model's own
// vocabulary, but it tracks it closely enough to size chunks against a
// model window, which character counts do not (CJK vs Latin differ ~3x).
type Tokenizer interface {
	Count(s string) int
}

type tiktokenCounter struct{ enc *tiktoken.Tiktoken }

// NewTokenizer loads cl100k_base from BPE ranks embedded in the binary, so it
// works offline and inside a scratch container.
func NewTokenizer() (Tokenizer, error) {
	tiktoken.SetBpeLoader(tiktokenloader.NewOfflineLoader())
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return nil, fmt.Errorf("load cl100k_base: %w", err)
	}
	return tiktokenCounter{enc}, nil
}

func (t tiktokenCounter) Count(s string) int { return len(t.enc.EncodeOrdinary(s)) }
