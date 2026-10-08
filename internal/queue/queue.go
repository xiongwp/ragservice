// Package queue is a small durable job queue on a MySQL table.
//
// Jobs are inserted inside the caller's transaction (so a document row and
// the job that indexes it commit together), claimed with
// SELECT ... FOR UPDATE SKIP LOCKED, retried with backoff, and reclaimed
// if a worker dies holding one.
package queue

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"sync"
	"time"

	"rag/internal/db"
)

type Job struct {
	ID          int64
	Kind        string
	Args        json.RawMessage
	Attempt     int // 1 on the first run
	MaxAttempts int
}

type Handler func(ctx context.Context, job *Job) error

type InsertOpts struct {
	// UniqueKey, if set, allows at most one queued-or-running job per key.
	UniqueKey   string
	MaxAttempts int // default 3
}

// ErrDuplicate: a job with the same UniqueKey is already queued or running.
var ErrDuplicate = errors.New("an equivalent job is already queued or running")

type cancelError struct{ err error }

func (e cancelError) Error() string { return e.err.Error() }
func (e cancelError) Unwrap() error { return e.err }

// Cancel marks a failure as permanent: the job is not retried.
func Cancel(err error) error { return cancelError{err} }

// Insert enqueues a job inside tx. Call Queue.Notify after commit to start
// it without waiting for the next poll.
func Insert(ctx context.Context, tx *sql.Tx, kind string, args any, opts InsertOpts) error {
	raw, err := json.Marshal(args)
	if err != nil {
		return err
	}
	if opts.MaxAttempts <= 0 {
		opts.MaxAttempts = 3
	}
	var key any
	if opts.UniqueKey != "" {
		key = opts.UniqueKey
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO jobs (kind, args, unique_key, max_attempts) VALUES (?, ?, ?, ?)`,
		kind, raw, key, opts.MaxAttempts)
	if db.IsDuplicate(err, "jobs_live_key") {
		return ErrDuplicate
	}
	return err
}

type registration struct {
	handler Handler
	timeout time.Duration
}

type Queue struct {
	db       *sql.DB
	workers  int
	log      *slog.Logger
	handlers map[string]registration
	wake     chan struct{}

	PollInterval  time.Duration
	RetryBackoff  func(attempt int) time.Duration // default Backoff
	LeaseGrace    time.Duration                   // added to a handler's timeout; default 1 min
	MaintainEvery time.Duration                   // reclaim/prune interval; default 30 s

	stopFetch context.CancelFunc
	stopJobs  context.CancelFunc
	wg        sync.WaitGroup
}

func New(sqlDB *sql.DB, workers int, log *slog.Logger) *Queue {
	return &Queue{
		db: sqlDB, workers: workers, log: log,
		handlers:      map[string]registration{},
		wake:          make(chan struct{}, 1),
		PollInterval:  time.Second,
		RetryBackoff:  Backoff,
		LeaseGrace:    time.Minute,
		MaintainEvery: 30 * time.Second,
	}
}

// Register must be called before Start.
func (q *Queue) Register(kind string, timeout time.Duration, h Handler) {
	q.handlers[kind] = registration{h, timeout}
}

// Notify wakes an idle worker.
func (q *Queue) Notify() {
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

func (q *Queue) Start(ctx context.Context) {
	fetchCtx, stopFetch := context.WithCancel(context.WithoutCancel(ctx))
	jobCtx, stopJobs := context.WithCancel(context.WithoutCancel(ctx))
	q.stopFetch, q.stopJobs = stopFetch, stopJobs
	for i := 0; i < q.workers; i++ {
		q.wg.Add(1)
		go func() {
			defer q.wg.Done()
			q.work(fetchCtx, jobCtx)
		}()
	}
	q.wg.Add(1)
	go func() {
		defer q.wg.Done()
		q.maintain(fetchCtx)
	}()
}

// Stop stops claiming jobs and waits for running ones until ctx expires,
// then cancels them; the rescuer requeues anything cut short.
func (q *Queue) Stop(ctx context.Context) error {
	q.stopFetch()
	done := make(chan struct{})
	go func() { q.wg.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		q.stopJobs()
		<-done
		return ctx.Err()
	}
}

func (q *Queue) work(fetchCtx, jobCtx context.Context) {
	for {
		if fetchCtx.Err() != nil {
			return
		}
		job, timeout, err := q.claim(fetchCtx)
		if err != nil && fetchCtx.Err() == nil {
			q.log.Error("claim job", "err", err)
		}
		if job == nil {
			select {
			case <-fetchCtx.Done():
				return
			case <-q.wake:
			case <-time.After(q.PollInterval):
			}
			continue
		}
		q.run(jobCtx, job, timeout)
	}
}

func (q *Queue) claim(ctx context.Context) (*Job, time.Duration, error) {
	var job *Job
	var timeout time.Duration
	err := db.Tx(ctx, q.db, func(tx *sql.Tx) error {
		var j Job
		err := tx.QueryRowContext(ctx, `SELECT id, kind, args, attempt, max_attempts FROM jobs
			WHERE state = 'available' AND run_at <= NOW(6)
			ORDER BY run_at, id LIMIT 1 FOR UPDATE SKIP LOCKED`).
			Scan(&j.ID, &j.Kind, &j.Args, &j.Attempt, &j.MaxAttempts)
		if errors.Is(err, sql.ErrNoRows) {
			return nil
		}
		if err != nil {
			return err
		}
		reg, ok := q.handlers[j.Kind]
		if !ok {
			_, err := tx.ExecContext(ctx, `UPDATE jobs SET state = 'cancelled', last_error = ? WHERE id = ?`,
				"no handler registered for kind "+j.Kind, j.ID)
			return err
		}
		timeout = reg.timeout
		// The lease outlives the handler timeout so a slow-but-alive job is
		// not reclaimed while it is still running.
		lease := (timeout + q.LeaseGrace).Microseconds()
		if _, err := tx.ExecContext(ctx, `UPDATE jobs SET state = 'running', attempt = attempt + 1,
			locked_until = NOW(6) + INTERVAL ? MICROSECOND WHERE id = ?`, lease, j.ID); err != nil {
			return err
		}
		j.Attempt++
		job = &j
		return nil
	})
	return job, timeout, err
}

func (q *Queue) run(ctx context.Context, job *Job, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	err := func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("panic: %v\n%s", r, debug.Stack())
			}
		}()
		return q.handlers[job.Kind].handler(ctx, job)
	}()

	finish := context.WithoutCancel(ctx)
	var res sql.Result
	var uerr error
	var ce cancelError
	switch {
	case err == nil:
		res, uerr = q.db.ExecContext(finish, `UPDATE jobs SET state = 'completed', locked_until = NULL, last_error = NULL
			WHERE id = ? AND state = 'running' AND attempt = ?`, job.ID, job.Attempt)
	case errors.As(err, &ce):
		res, uerr = q.db.ExecContext(finish, `UPDATE jobs SET state = 'cancelled', locked_until = NULL, last_error = ?
			WHERE id = ? AND state = 'running' AND attempt = ?`, err.Error(), job.ID, job.Attempt)
	case job.Attempt >= job.MaxAttempts:
		res, uerr = q.db.ExecContext(finish, `UPDATE jobs SET state = 'failed', locked_until = NULL, last_error = ?
			WHERE id = ? AND state = 'running' AND attempt = ?`, err.Error(), job.ID, job.Attempt)
	default:
		res, uerr = q.db.ExecContext(finish, `UPDATE jobs SET state = 'available', locked_until = NULL, last_error = ?,
			run_at = NOW(6) + INTERVAL ? MICROSECOND WHERE id = ? AND state = 'running' AND attempt = ?`,
			err.Error(), q.RetryBackoff(job.Attempt).Microseconds(), job.ID, job.Attempt)
	}
	if uerr != nil {
		q.log.Error("record job result", "job_id", job.ID, "err", uerr)
		return
	}
	if n, _ := res.RowsAffected(); n == 0 {
		q.log.Warn("job was reclaimed while running", "job_id", job.ID, "kind", job.Kind)
	}
}

// Backoff before retry number attempt+1: 10s, 40s, 90s, ... capped at 10 min.
func Backoff(attempt int) time.Duration {
	return min(time.Duration(attempt*attempt)*10*time.Second, 10*time.Minute)
}

// maintain requeues jobs whose worker died and prunes old finished jobs.
func (q *Queue) maintain(ctx context.Context) {
	tick := time.NewTicker(q.MaintainEvery)
	defer tick.Stop()
	for {
		if res, err := q.db.ExecContext(ctx, `UPDATE jobs
			SET state = IF(attempt >= max_attempts, 'failed', 'available'), locked_until = NULL,
			    last_error = 'lease expired: worker stopped or job exceeded its timeout'
			WHERE state = 'running' AND locked_until < NOW(6)`); err == nil {
			if n, _ := res.RowsAffected(); n > 0 {
				q.log.Warn("reclaimed expired jobs", "count", n)
			}
		} else if ctx.Err() == nil {
			q.log.Error("reclaim jobs", "err", err)
		}
		if _, err := q.db.ExecContext(ctx, `DELETE FROM jobs WHERE
			(state IN ('completed', 'cancelled') AND updated_at < NOW(6) - INTERVAL 1 DAY) OR
			(state = 'failed' AND updated_at < NOW(6) - INTERVAL 7 DAY) LIMIT 1000`); err != nil && ctx.Err() == nil {
			q.log.Error("prune jobs", "err", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}
