package queue_test

// Needs MySQL: RAG_TEST_MYSQL_DSN='rag:rag@tcp(localhost:33306)/rag_test'.
// Uses its own database (<name>_queue) so it can run beside the e2e test.

import (
	"context"
	"database/sql"
	"errors"
	"io"
	"log/slog"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-sql-driver/mysql"

	"rag/internal/db"
	"rag/internal/queue"
)

func setup(t *testing.T) *sql.DB {
	t.Helper()
	dsn := os.Getenv("RAG_TEST_MYSQL_DSN")
	if dsn == "" {
		t.Skip("RAG_TEST_MYSQL_DSN not set")
	}
	cfg, err := mysql.ParseDSN(dsn)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := db.Open(t.Context(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	name := cfg.DBName + "_queue"
	for _, q := range []string{"DROP DATABASE IF EXISTS `" + name + "`", "CREATE DATABASE `" + name + "`"} {
		if _, err := admin.ExecContext(t.Context(), q); err != nil {
			t.Fatal(err)
		}
	}
	admin.Close()
	cfg.DBName = name
	if err := db.Migrate(t.Context(), cfg.FormatDSN(), slog.New(slog.NewTextHandler(io.Discard, nil))); err != nil {
		t.Fatal(err)
	}
	sqlDB, err := db.Open(t.Context(), cfg.FormatDSN())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sqlDB.Close() })
	return sqlDB
}

func newQueue(sqlDB *sql.DB) *queue.Queue {
	q := queue.New(sqlDB, 3, slog.New(slog.NewTextHandler(io.Discard, nil)))
	q.PollInterval = 20 * time.Millisecond
	q.RetryBackoff = func(int) time.Duration { return 10 * time.Millisecond }
	return q
}

func insert(t *testing.T, sqlDB *sql.DB, kind string, opts queue.InsertOpts) error {
	t.Helper()
	return db.Tx(t.Context(), sqlDB, func(tx *sql.Tx) error {
		return queue.Insert(t.Context(), tx, kind, map[string]string{"k": "v"}, opts)
	})
}

type jobRow struct {
	state, lastError string
	attempt          int
}

func waitState(t *testing.T, sqlDB *sql.DB, kind, want string) jobRow {
	t.Helper()
	var r jobRow
	var lastErr sql.NullString
	for i := 0; i < 200; i++ {
		err := sqlDB.QueryRowContext(t.Context(), `SELECT state, attempt, last_error FROM jobs WHERE kind = ? ORDER BY id DESC LIMIT 1`, kind).
			Scan(&r.state, &r.attempt, &lastErr)
		if err == nil && r.state == want {
			r.lastError = lastErr.String
			return r
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("%s: state %q (attempt %d), want %q", kind, r.state, r.attempt, want)
	return r
}

func TestRetryThenSucceed(t *testing.T) {
	sqlDB := setup(t)
	q := newQueue(sqlDB)
	var calls atomic.Int32
	q.Register("flaky", time.Minute, func(ctx context.Context, job *queue.Job) error {
		if calls.Add(1) < 3 {
			return errors.New("transient")
		}
		if job.Attempt != 3 {
			t.Errorf("attempt = %d, want 3", job.Attempt)
		}
		return nil
	})
	q.Register("broken", time.Minute, func(context.Context, *queue.Job) error { return errors.New("always") })
	q.Register("hopeless", time.Minute, func(context.Context, *queue.Job) error { return queue.Cancel(errors.New("bad input")) })
	q.Register("panics", time.Minute, func(context.Context, *queue.Job) error { panic("boom") })
	q.Start(t.Context())
	defer q.Stop(context.Background())

	for _, kind := range []string{"flaky", "broken", "hopeless", "panics"} {
		if err := insert(t, sqlDB, kind, queue.InsertOpts{MaxAttempts: 3}); err != nil {
			t.Fatal(err)
		}
	}
	q.Notify()

	if r := waitState(t, sqlDB, "flaky", "completed"); r.attempt != 3 {
		t.Errorf("flaky completed after %d attempts", r.attempt)
	}
	if r := waitState(t, sqlDB, "broken", "failed"); r.attempt != 3 || r.lastError != "always" {
		t.Errorf("broken = %+v", r)
	}
	if r := waitState(t, sqlDB, "hopeless", "cancelled"); r.attempt != 1 {
		t.Errorf("cancelled job was retried: %+v", r)
	}
	if r := waitState(t, sqlDB, "panics", "failed"); r.attempt != 3 {
		t.Errorf("panicking job = %+v", r)
	}
}

func TestUniqueKey(t *testing.T) {
	sqlDB := setup(t)
	opts := queue.InsertOpts{UniqueKey: "doc:1"}
	if err := insert(t, sqlDB, "x", opts); err != nil {
		t.Fatal(err)
	}
	if err := insert(t, sqlDB, "x", opts); !errors.Is(err, queue.ErrDuplicate) {
		t.Fatalf("second live job: err = %v, want ErrDuplicate", err)
	}
	// Once the first finishes, the key is free again.
	q := newQueue(sqlDB)
	q.Register("x", time.Minute, func(context.Context, *queue.Job) error { return nil })
	q.Start(t.Context())
	waitState(t, sqlDB, "x", "completed")
	q.Stop(context.Background())
	if err := insert(t, sqlDB, "x", opts); err != nil {
		t.Errorf("after completion: %v", err)
	}
}

func TestExpiredLeaseIsReclaimed(t *testing.T) {
	sqlDB := setup(t)
	// Simulate a worker that died holding the job.
	if _, err := sqlDB.ExecContext(t.Context(), `INSERT INTO jobs (kind, args, state, attempt, max_attempts, locked_until)
		VALUES ('orphan', '{}', 'running', 1, 3, NOW(6) - INTERVAL 1 SECOND)`); err != nil {
		t.Fatal(err)
	}
	q := newQueue(sqlDB)
	q.MaintainEvery = 20 * time.Millisecond
	q.Register("orphan", time.Minute, func(context.Context, *queue.Job) error { return nil })
	q.Start(t.Context())
	defer q.Stop(context.Background())
	if r := waitState(t, sqlDB, "orphan", "completed"); r.attempt != 2 {
		t.Errorf("reclaimed job = %+v", r)
	}
}

func TestHandlerTimeout(t *testing.T) {
	sqlDB := setup(t)
	q := newQueue(sqlDB)
	q.Register("slow", 50*time.Millisecond, func(ctx context.Context, _ *queue.Job) error {
		<-ctx.Done()
		return ctx.Err()
	})
	q.Start(t.Context())
	defer q.Stop(context.Background())
	if err := insert(t, sqlDB, "slow", queue.InsertOpts{MaxAttempts: 1}); err != nil {
		t.Fatal(err)
	}
	q.Notify()
	if r := waitState(t, sqlDB, "slow", "failed"); r.lastError != context.DeadlineExceeded.Error() {
		t.Errorf("slow = %+v", r)
	}
}
