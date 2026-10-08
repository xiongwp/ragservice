package retrieve

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func cond(t *testing.T, js string) *MetadataCondition {
	t.Helper()
	var mc MetadataCondition
	if err := json.Unmarshal([]byte(js), &mc); err != nil {
		t.Fatal(err)
	}
	return &mc
}

func noResolve(context.Context, string, string) ([]string, error) {
	return nil, errors.New("resolver should not be called")
}

func TestBuildFilterShape(t *testing.T) {
	f, err := BuildFilter(t.Context(), cond(t, `{"logical_operator":"or","conditions":[
		{"name":"dept","comparison_operator":"is","value":"hr"},
		{"name":"year","comparison_operator":"is","value":"2024"},
		{"name":"score","comparison_operator":"≥","value":3.5},
		{"name":"note","comparison_operator":"not empty"}]}`), noResolve)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Should) != 4 || len(f.Must) != 0 {
		t.Fatalf("or should map to 4 Should conditions, got %v", f)
	}
	// "hr": plain keyword match on the nested metadata key.
	if fc := f.Should[0].GetField(); fc == nil || fc.Key != "metadata.dept" || fc.GetMatch().GetKeyword() != "hr" {
		t.Errorf("is hr = %v", f.Should[0])
	}
	// "2024": keyword OR number, so a numeric metadata value matches too.
	alts := f.Should[1].GetFilter().GetShould()
	if len(alts) != 2 || alts[0].GetField().GetMatch().GetKeyword() != "2024" ||
		alts[1].GetField().GetRange().GetGte() != 2024 || alts[1].GetField().GetRange().GetLte() != 2024 {
		t.Errorf("is 2024 = %v", f.Should[1])
	}
	if r := f.Should[2].GetField().GetRange(); r == nil || r.GetGte() != 3.5 || r.Lt != nil {
		t.Errorf("≥ 3.5 = %v", f.Should[2])
	}
	if len(f.Should[3].GetFilter().GetMustNot()) != 1 {
		t.Errorf("not empty should be a MustNot, got %v", f.Should[3])
	}
}

func TestPrefixGoesThroughResolver(t *testing.T) {
	var gotField, gotPattern string
	resolve := func(_ context.Context, field, pattern string) ([]string, error) {
		gotField, gotPattern = field, pattern
		return []string{"doc-1", "doc-2"}, nil
	}
	f, err := BuildFilter(t.Context(), cond(t, `{"conditions":[{"name":"code","comparison_operator":"start with","value":"50%_x"}]}`), resolve)
	if err != nil {
		t.Fatal(err)
	}
	if gotField != "code" || gotPattern != "50!%!_x%" {
		t.Errorf("resolver got %q %q", gotField, gotPattern)
	}
	fc := f.Must[0].GetField()
	if fc.Key != "document_id" || len(fc.GetMatch().GetKeywords().GetStrings()) != 2 {
		t.Errorf("start with = %v", f.Must[0])
	}

	none := func(context.Context, string, string) ([]string, error) { return nil, nil }
	f, _ = BuildFilter(t.Context(), cond(t, `{"conditions":[{"name":"code","comparison_operator":"end with","value":"z"}]}`), none)
	if f.Must[0].GetField().GetMatch().GetKeyword() != "-none-" {
		t.Errorf("no matching documents must match nothing, got %v", f.Must[0])
	}
}

func TestBuildFilterEmpty(t *testing.T) {
	for _, mc := range []*MetadataCondition{nil, {LogicalOperator: "and"}} {
		f, err := BuildFilter(t.Context(), mc, noResolve)
		if f != nil || err != nil {
			t.Errorf("empty condition should be no filter, got %v %v", f, err)
		}
	}
}

func TestBuildFilterErrors(t *testing.T) {
	for _, js := range []string{
		`{"conditions":[{"name":"x","comparison_operator":"like","value":"a"}]}`,
		`{"conditions":[{"name":"x","comparison_operator":">","value":"abc"}]}`,
		`{"conditions":[{"name":"","comparison_operator":"is","value":"a"}]}`,
		`{"conditions":[{"name":"a.b","comparison_operator":"is","value":"a"}]}`,
		`{"logical_operator":"xor","conditions":[{"name":"x","comparison_operator":"is","value":"a"}]}`,
		`{"conditions":[{"name":"x","comparison_operator":"before","value":"yesterday"}]}`,
	} {
		_, err := BuildFilter(t.Context(), cond(t, js), noResolve)
		var fe *FilterError
		if !errors.As(err, &fe) {
			t.Errorf("%s: want FilterError, got %v", js, err)
		}
	}
}
