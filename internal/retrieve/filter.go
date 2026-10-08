package retrieve

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/protobuf/types/known/timestamppb"

	"rag/internal/store"
	"rag/internal/vector"
)

// MetadataCondition is Dify's metadata_condition, sent as-is on /retrieval.
type MetadataCondition struct {
	LogicalOperator string      `json:"logical_operator"`
	Conditions      []Condition `json:"conditions"`
}

type Condition struct {
	Name               string          `json:"name"`
	ComparisonOperator string          `json:"comparison_operator"`
	Value              json.RawMessage `json:"value"`
}

// FilterError is a malformed condition: the caller's fault, reported as 400.
type FilterError struct{ msg string }

func (e *FilterError) Error() string { return e.msg }

func filterErr(format string, a ...any) error { return &FilterError{fmt.Sprintf(format, a...)} }

// Resolver returns ids of documents whose metadata field, read as text,
// matches a LIKE pattern (escaped with store.EscapeLike). It serves the
// substring operators, which Qdrant cannot evaluate on arbitrary keys.
type Resolver func(ctx context.Context, field, likePattern string) ([]string, error)

// BuildFilter turns Dify's metadata condition into a Qdrant filter over the
// "metadata" payload object. An empty condition yields nil (no filter).
func BuildFilter(ctx context.Context, mc *MetadataCondition, resolve Resolver) (*qdrant.Filter, error) {
	if mc == nil || len(mc.Conditions) == 0 {
		return nil, nil
	}
	var or bool
	switch strings.ToLower(mc.LogicalOperator) {
	case "", "and":
	case "or":
		or = true
	default:
		return nil, filterErr("logical_operator must be \"and\" or \"or\", got %q", mc.LogicalOperator)
	}

	var conds []*qdrant.Condition
	for _, c := range mc.Conditions {
		cond, err := buildCondition(ctx, c, resolve)
		if err != nil {
			return nil, err
		}
		conds = append(conds, cond)
	}
	if or {
		return &qdrant.Filter{Should: conds}, nil
	}
	return &qdrant.Filter{Must: conds}, nil
}

func buildCondition(ctx context.Context, c Condition, resolve Resolver) (*qdrant.Condition, error) {
	if c.Name == "" {
		return nil, filterErr("condition is missing name")
	}
	if strings.ContainsAny(c.Name, ".[]\"") {
		return nil, filterErr("metadata name %q may not contain . [ ] or \"", c.Name)
	}
	key := vector.FieldMetadata + "." + c.Name
	op := c.ComparisonOperator
	wrap := func(err error) error { return filterErr("%s %q: %v", c.Name, op, err) }

	switch op {
	case "is", "is not", "contains", "not contains", "start with", "end with":
		v, err := stringValue(c.Value)
		if err != nil {
			return nil, wrap(err)
		}
		switch op {
		case "is":
			return anyOf(equals(key, v)...), nil
		case "is not":
			return not(anyOf(equals(key, v)...)), nil
		}
		// Substring tests have no Qdrant equivalent on unindexed keys (MatchText
		// needs a full-text index and tokenizes), so MySQL finds the documents.
		esc := store.EscapeLike(v)
		pattern := map[string]string{
			"contains": "%" + esc + "%", "not contains": "%" + esc + "%",
			"start with": esc + "%", "end with": "%" + esc,
		}[op]
		ids, err := resolve(ctx, c.Name, pattern)
		if err != nil {
			return nil, err
		}
		docs := qdrant.NewMatchKeyword(vector.FieldDocument, "-none-") // matches nothing
		if len(ids) > 0 {
			docs = qdrant.NewMatchKeywords(vector.FieldDocument, ids...)
		}
		if op == "not contains" {
			return not(docs), nil
		}
		return docs, nil
	case "empty", "not empty":
		empty := anyOf(qdrant.NewIsEmpty(key), qdrant.NewMatchKeyword(key, ""))
		if op == "empty" {
			return empty, nil
		}
		return not(empty), nil
	case "in", "not in":
		vs, err := stringList(c.Value)
		if err != nil {
			return nil, wrap(err)
		}
		var alts []*qdrant.Condition
		for _, v := range vs {
			alts = append(alts, equals(key, v)...)
		}
		if len(alts) == 0 {
			alts = []*qdrant.Condition{qdrant.NewMatchKeyword(vector.FieldDocument, "-none-")}
		}
		if op == "in" {
			return anyOf(alts...), nil
		}
		return not(anyOf(alts...)), nil
	case "=", "≠", ">", "<", "≥", "≤":
		n, err := numberValue(c.Value)
		if err != nil {
			return nil, wrap(err)
		}
		switch op {
		case "=":
			return qdrant.NewRange(key, &qdrant.Range{Gte: &n, Lte: &n}), nil
		case "≠": // the field must exist and differ
			return anyOf(qdrant.NewRange(key, &qdrant.Range{Lt: &n}), qdrant.NewRange(key, &qdrant.Range{Gt: &n})), nil
		case ">":
			return qdrant.NewRange(key, &qdrant.Range{Gt: &n}), nil
		case "<":
			return qdrant.NewRange(key, &qdrant.Range{Lt: &n}), nil
		case "≥":
			return qdrant.NewRange(key, &qdrant.Range{Gte: &n}), nil
		default:
			return qdrant.NewRange(key, &qdrant.Range{Lte: &n}), nil
		}
	case "before", "after":
		secs, err := epochValue(c.Value)
		if err != nil {
			return nil, wrap(err)
		}
		// The field may hold unix seconds or an RFC 3339 string; accept either.
		ts := timestamppb.New(time.Unix(int64(secs), 0))
		if op == "before" {
			return anyOf(qdrant.NewRange(key, &qdrant.Range{Lt: &secs}),
				qdrant.NewDatetimeRange(key, &qdrant.DatetimeRange{Lt: ts})), nil
		}
		return anyOf(qdrant.NewRange(key, &qdrant.Range{Gt: &secs}),
			qdrant.NewDatetimeRange(key, &qdrant.DatetimeRange{Gt: ts})), nil
	}
	return nil, filterErr("unsupported comparison_operator %q", op)
}

// equals matches the field against v as a keyword, and also as the typed
// JSON value v spells (number, boolean), so {"year": 2024} matches "2024".
// Numbers use a closed range rather than an integer match: JSON numbers
// reach Qdrant as doubles, which an integer match never equals.
func equals(key, v string) []*qdrant.Condition {
	alts := []*qdrant.Condition{qdrant.NewMatchKeyword(key, v)}
	if f, err := strconv.ParseFloat(v, 64); err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) {
		alts = append(alts, qdrant.NewRange(key, &qdrant.Range{Gte: &f, Lte: &f}))
	}
	if v == "true" || v == "false" {
		alts = append(alts, qdrant.NewMatchBool(key, v == "true"))
	}
	return alts
}

func anyOf(conds ...*qdrant.Condition) *qdrant.Condition {
	if len(conds) == 1 {
		return conds[0]
	}
	return qdrant.NewFilterAsCondition(&qdrant.Filter{Should: conds})
}

func not(c *qdrant.Condition) *qdrant.Condition {
	return qdrant.NewFilterAsCondition(&qdrant.Filter{MustNot: []*qdrant.Condition{c}})
}

func stringValue(raw json.RawMessage) (string, error) {
	var v any
	if err := json.Unmarshal(orNull(raw), &v); err != nil {
		return "", err
	}
	switch x := v.(type) {
	case string:
		return x, nil
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64), nil
	case bool:
		return strconv.FormatBool(x), nil
	}
	return "", fmt.Errorf("value must be a string")
}

func stringList(raw json.RawMessage) ([]string, error) {
	var list []string
	if err := json.Unmarshal(orNull(raw), &list); err == nil && list != nil {
		return list, nil
	}
	s, err := stringValue(raw)
	if err != nil {
		return nil, fmt.Errorf("value must be an array of strings")
	}
	return []string{s}, nil
}

func numberValue(raw json.RawMessage) (float64, error) {
	var v any
	if err := json.Unmarshal(orNull(raw), &v); err != nil {
		return 0, err
	}
	switch x := v.(type) {
	case float64:
		return x, nil
	case string:
		if f, err := strconv.ParseFloat(strings.TrimSpace(x), 64); err == nil {
			return f, nil
		}
	}
	return 0, fmt.Errorf("value must be a number")
}

var timeLayouts = []string{time.RFC3339, "2006-01-02 15:04:05", "2006-01-02T15:04:05", "2006-01-02"}

// epochValue accepts unix seconds or a date/time string (UTC if no zone).
func epochValue(raw json.RawMessage) (float64, error) {
	if n, err := numberValue(raw); err == nil {
		return n, nil
	}
	s, err := stringValue(raw)
	if err == nil {
		for _, layout := range timeLayouts {
			if t, err := time.Parse(layout, strings.TrimSpace(s)); err == nil {
				return float64(t.Unix()), nil
			}
		}
	}
	return 0, fmt.Errorf("value must be unix seconds or a date like 2025-01-31")
}

func orNull(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("null")
	}
	return raw
}
