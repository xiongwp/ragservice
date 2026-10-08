package retrieve

import (
	"testing"

	"github.com/google/uuid"

	"rag/internal/vector"
)

func TestFuse(t *testing.T) {
	a, b, c := uuid.New(), uuid.New(), uuid.New()
	dense := []vector.Hit{{ID: a, Score: 0.9}, {ID: b, Score: 0.8}}
	sparse := []vector.Hit{{ID: b, Score: 12}, {ID: c, Score: 7}}

	got := fuse([][]vector.Hit{dense}, [][]vector.Hit{sparse})
	if len(got) != 3 || got[0].id != b {
		t.Fatalf("b is in both lists and must win: %+v", got)
	}
	// b: 1/62 + 1/61, normalised by 2/61.
	if want := (1.0/62 + 1.0/61) / (2.0 / 61); abs(got[0].score-want) > 1e-12 {
		t.Errorf("b score = %v, want %v", got[0].score, want)
	}
	if got[0].vectorRank != 2 || got[0].keywordRank != 1 {
		t.Errorf("b ranks = %d/%d", got[0].vectorRank, got[0].keywordRank)
	}
	// a (dense #1) and c (sparse #2): a ranks higher.
	if got[1].id != a || got[2].id != c {
		t.Errorf("order = %v %v %v", got[0].id, got[1].id, got[2].id)
	}

	// With only one non-empty list, its top hit scores exactly 1.
	if got := fuse([][]vector.Hit{dense}, [][]vector.Hit{nil}); got[0].score != 1 {
		t.Errorf("single-list top score = %v", got[0].score)
	}
	// A single vector list reports cosine similarity.
	if got := fuse([][]vector.Hit{dense}, nil); len(got) != 2 || abs(got[0].score-0.9) > 1e-6 {
		t.Errorf("dense mode = %+v", got)
	}
	// Several vector lists (query variants) keep the best rank per kind.
	got = fuse([][]vector.Hit{dense, {{ID: b}, {ID: a}}}, nil)
	if got[0].vectorRank != 1 || got[1].vectorRank != 1 {
		t.Errorf("best vector ranks = %d, %d", got[0].vectorRank, got[1].vectorRank)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
