// Package db opens MySQL and applies schema migrations.
package db

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/go-sql-driver/mysql"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// config parses dsn and pins the settings the code relies on: UTC
// timestamps on both sides, parsed DATETIMEs and utf8mb4.
func config(dsn string, multiStatements bool) (*mysql.Config, error) {
	cfg, err := mysql.ParseDSN(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse RAG_MYSQL_DSN: %w", err)
	}
	cfg.ParseTime = true
	cfg.Loc = time.UTC
	cfg.Collation = "utf8mb4_0900_ai_ci"
	cfg.MultiStatements = multiStatements
	if cfg.Params == nil {
		cfg.Params = map[string]string{}
	}
	cfg.Params["time_zone"] = "'+00:00'"
	return cfg, nil
}

func Open(ctx context.Context, dsn string) (*sql.DB, error) {
	cfg, err := config(dsn, false)
	if err != nil {
		return nil, err
	}
	conn, err := mysql.NewConnector(cfg)
	if err != nil {
		return nil, err
	}
	db := sql.OpenDB(conn)
	db.SetMaxOpenConns(50)
	db.SetMaxIdleConns(25)
	db.SetConnMaxLifetime(30 * time.Minute)
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping mysql: %w", err)
	}
	return db, nil
}

// Migrate applies the embedded migrations in order. MySQL DDL is not
// transactional, so every statement is written to be safely re-runnable.
// A named lock serialises concurrent starts of several replicas.
func Migrate(ctx context.Context, dsn string, log *slog.Logger) error {
	cfg, err := config(dsn, true)
	if err != nil {
		return err
	}
	connector, err := mysql.NewConnector(cfg)
	if err != nil {
		return err
	}
	db := sql.OpenDB(connector)
	defer db.Close()
	conn, err := db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("connect mysql: %w", err)
	}
	defer conn.Close()

	var got sql.NullInt64
	if err := conn.QueryRowContext(ctx, "SELECT GET_LOCK('rag_migrate', 120)").Scan(&got); err != nil {
		return err
	}
	if got.Int64 != 1 {
		return errors.New("timed out waiting for the migration lock")
	}
	defer conn.ExecContext(context.WithoutCancel(ctx), "SELECT RELEASE_LOCK('rag_migrate')") //nolint:errcheck

	if _, err := conn.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS rag_schema_migrations (
		version VARCHAR(191) NOT NULL PRIMARY KEY,
		applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6))`); err != nil {
		return err
	}
	names, err := fs.Glob(migrationFS, "migrations/*.sql")
	if err != nil {
		return err
	}
	sort.Strings(names)
	for _, name := range names {
		version := strings.TrimSuffix(strings.TrimPrefix(name, "migrations/"), ".sql")
		var n int
		if err := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM rag_schema_migrations WHERE version = ?", version).Scan(&n); err != nil {
			return err
		}
		if n > 0 {
			continue
		}
		body, err := migrationFS.ReadFile(name)
		if err != nil {
			return err
		}
		if _, err := conn.ExecContext(ctx, string(body)); err != nil {
			return fmt.Errorf("migration %s: %w", version, err)
		}
		if _, err := conn.ExecContext(ctx, "INSERT INTO rag_schema_migrations (version) VALUES (?)", version); err != nil {
			return err
		}
		log.Info("applied migration", "version", version)
	}
	return nil
}

// WaitReady blocks until MySQL accepts connections or timeout passes, so
// the service can start alongside its database instead of crash-looping.
func WaitReady(ctx context.Context, dsn string, timeout time.Duration, log *slog.Logger) error {
	cfg, err := config(dsn, false)
	if err != nil {
		return err
	}
	connector, err := mysql.NewConnector(cfg)
	if err != nil {
		return err
	}
	sqlDB := sql.OpenDB(connector)
	defer sqlDB.Close()
	deadline := time.Now().Add(timeout)
	for {
		pctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		err := sqlDB.PingContext(pctx)
		cancel()
		if err == nil {
			return nil
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			return fmt.Errorf("mysql not reachable after %s: %w", timeout, err)
		}
		log.Info("waiting for mysql", "err", err)
		select {
		case <-time.After(2 * time.Second):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Tx runs fn in a transaction, committing on nil and rolling back otherwise.
func Tx(ctx context.Context, db *sql.DB, fn func(*sql.Tx) error) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		_ = tx.Rollback()
		return err
	}
	return tx.Commit()
}

// IsDuplicate reports a unique-key violation, optionally on a given index.
func IsDuplicate(err error, index string) bool {
	var me *mysql.MySQLError
	if !errors.As(err, &me) || me.Number != 1062 {
		return false
	}
	return index == "" || strings.Contains(me.Message, index)
}
