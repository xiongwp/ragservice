// Package auth resolves bearer tokens to a tenant and a role, and manages
// tenants and API keys.
//
// Roles, each including the ones below it:
//
//	root    keys from RAG_API_KEYS; manage tenants; act on any tenant via X-Tenant-ID
//	admin   manage the tenant's API keys
//	editor  create, change and delete datasets and documents
//	reader  retrieve and read
package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type Role int

const (
	Reader Role = iota + 1
	Editor
	Admin
	Root
)

func (r Role) String() string {
	return map[Role]string{Reader: "reader", Editor: "editor", Admin: "admin", Root: "root"}[r]
}

func ParseRole(s string) (Role, bool) {
	r, ok := map[string]Role{"reader": Reader, "editor": Editor, "admin": Admin}[s]
	return r, ok
}

type Tenant struct {
	ID           uuid.UUID `json:"id"`
	Name         string    `json:"name"`
	RateLimitRPS float64   `json:"rate_limit_rps"`
	MaxDatasets  int       `json:"max_datasets"`
	MaxDocuments int       `json:"max_documents"`
	MaxStorageMB int       `json:"max_storage_mb"`
	CreatedAt    time.Time `json:"created_at"`
}

// Principal is who is calling.
type Principal struct {
	Tenant  Tenant
	Role    Role
	KeyID   string // empty for root keys
	KeyName string
}

type Key struct {
	ID         uuid.UUID  `json:"id"`
	TenantID   uuid.UUID  `json:"tenant_id"`
	Name       string     `json:"name"`
	Prefix     string     `json:"prefix"`
	Role       string     `json:"role"`
	CreatedAt  time.Time  `json:"created_at"`
	LastUsedAt *time.Time `json:"last_used_at,omitempty"`
	RevokedAt  *time.Time `json:"revoked_at,omitempty"`
}

var (
	ErrUnauthorized = errors.New("invalid API key")
	ErrNotFound     = errors.New("not found")
)

type Authenticator struct {
	db            *sql.DB
	rootKeys      []string
	defaultTenant uuid.UUID

	mu    sync.Mutex
	cache map[string]cached // key hash -> principal
	ttl   time.Duration
}

type cached struct {
	p       Principal
	expires time.Time
}

func New(sqlDB *sql.DB, rootKeys []string, defaultTenant uuid.UUID) *Authenticator {
	return &Authenticator{db: sqlDB, rootKeys: rootKeys, defaultTenant: defaultTenant,
		cache: map[string]cached{}, ttl: 30 * time.Second}
}

// EnsureDefaultTenant creates the tenant root keys act on by default.
func (a *Authenticator) EnsureDefaultTenant(ctx context.Context) error {
	_, err := a.db.ExecContext(ctx, `INSERT IGNORE INTO tenants (id, name) VALUES (?, 'default')`, a.defaultTenant.String())
	return err
}

func hashKey(key string) []byte {
	h := sha256.Sum256([]byte(key))
	return h[:]
}

// Authenticate resolves a bearer token. tenantHeader (X-Tenant-ID) lets a
// root key act on a tenant other than the default one.
func (a *Authenticator) Authenticate(ctx context.Context, token, tenantHeader string) (*Principal, error) {
	if token == "" {
		return nil, ErrUnauthorized
	}
	root := 0
	for _, k := range a.rootKeys {
		root |= subtle.ConstantTimeCompare([]byte(token), []byte(k))
	}
	if root == 1 {
		id := a.defaultTenant
		if tenantHeader != "" {
			var err error
			if id, err = uuid.Parse(tenantHeader); err != nil {
				return nil, fmt.Errorf("%w: bad X-Tenant-ID", ErrUnauthorized)
			}
		}
		t, err := a.GetTenant(ctx, id)
		if errors.Is(err, ErrNotFound) {
			return nil, fmt.Errorf("%w: tenant %s does not exist", ErrUnauthorized, id)
		}
		if err != nil {
			return nil, err
		}
		return &Principal{Tenant: *t, Role: Root, KeyName: "root"}, nil
	}

	h := string(hashKey(token))
	a.mu.Lock()
	c, ok := a.cache[h]
	a.mu.Unlock()
	if ok && time.Now().Before(c.expires) {
		p := c.p
		return &p, nil
	}

	var p Principal
	var tid, kid, role string
	var lastUsed sql.NullTime
	err := a.db.QueryRowContext(ctx, `SELECT k.id, k.name, k.role, k.last_used_at,
			t.id, t.name, t.rate_limit_rps, t.max_datasets, t.max_documents, t.max_storage_mb, t.created_at
		FROM api_keys k JOIN tenants t ON t.id = k.tenant_id
		WHERE k.key_hash = ? AND k.revoked_at IS NULL`, []byte(h)).
		Scan(&kid, &p.KeyName, &role, &lastUsed, &tid, &p.Tenant.Name, &p.Tenant.RateLimitRPS,
			&p.Tenant.MaxDatasets, &p.Tenant.MaxDocuments, &p.Tenant.MaxStorageMB, &p.Tenant.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrUnauthorized
	}
	if err != nil {
		return nil, err
	}
	p.KeyID, p.Tenant.ID = kid, uuid.MustParse(tid)
	p.Role, _ = ParseRole(role)
	// last_used_at is advisory; write it at most once a minute per key.
	if !lastUsed.Valid || time.Since(lastUsed.Time) > time.Minute {
		_, _ = a.db.ExecContext(ctx, `UPDATE api_keys SET last_used_at = NOW(6) WHERE id = ?`, kid)
	}
	a.mu.Lock()
	a.cache[h] = cached{p, time.Now().Add(a.ttl)}
	a.mu.Unlock()
	return &p, nil
}

// Forget drops cached principals, so revocations and limit changes take
// effect at once on this replica (others catch up within the cache TTL).
func (a *Authenticator) Forget() {
	a.mu.Lock()
	a.cache = map[string]cached{}
	a.mu.Unlock()
}

// ---- tenants ----

const tenantCols = `id, name, rate_limit_rps, max_datasets, max_documents, max_storage_mb, created_at`

func scanTenant(row interface{ Scan(...any) error }) (*Tenant, error) {
	var t Tenant
	var id string
	err := row.Scan(&id, &t.Name, &t.RateLimitRPS, &t.MaxDatasets, &t.MaxDocuments, &t.MaxStorageMB, &t.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	t.ID = uuid.MustParse(id)
	return &t, nil
}

func (a *Authenticator) GetTenant(ctx context.Context, id uuid.UUID) (*Tenant, error) {
	return scanTenant(a.db.QueryRowContext(ctx, `SELECT `+tenantCols+` FROM tenants WHERE id = ?`, id.String()))
}

func (a *Authenticator) ListTenants(ctx context.Context) ([]*Tenant, error) {
	rows, err := a.db.QueryContext(ctx, `SELECT `+tenantCols+` FROM tenants ORDER BY created_at`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Tenant{}
	for rows.Next() {
		t, err := scanTenant(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func (a *Authenticator) SaveTenant(ctx context.Context, t *Tenant) error {
	if t.ID == uuid.Nil {
		t.ID = uuid.Must(uuid.NewV7())
	}
	_, err := a.db.ExecContext(ctx, `INSERT INTO tenants (id, name, rate_limit_rps, max_datasets, max_documents, max_storage_mb)
		VALUES (?, ?, ?, ?, ?, ?) ON DUPLICATE KEY UPDATE name = VALUES(name), rate_limit_rps = VALUES(rate_limit_rps),
		max_datasets = VALUES(max_datasets), max_documents = VALUES(max_documents), max_storage_mb = VALUES(max_storage_mb)`,
		t.ID.String(), t.Name, t.RateLimitRPS, t.MaxDatasets, t.MaxDocuments, t.MaxStorageMB)
	a.Forget()
	return err
}

// ---- keys ----

// CreateKey returns the new key in plain text; only its hash is stored.
func (a *Authenticator) CreateKey(ctx context.Context, tenant uuid.UUID, name string, role Role) (string, *Key, error) {
	raw := make([]byte, 24)
	if _, err := rand.Read(raw); err != nil {
		return "", nil, err
	}
	secret := "rag_" + base64.RawURLEncoding.EncodeToString(raw)
	k := &Key{ID: uuid.Must(uuid.NewV7()), TenantID: tenant, Name: name, Prefix: secret[:10], Role: role.String(),
		CreatedAt: time.Now().UTC()}
	_, err := a.db.ExecContext(ctx, `INSERT INTO api_keys (id, tenant_id, name, prefix, key_hash, role) VALUES (?, ?, ?, ?, ?, ?)`,
		k.ID.String(), tenant.String(), name, k.Prefix, hashKey(secret), k.Role)
	if err != nil {
		return "", nil, err
	}
	return secret, k, nil
}

func (a *Authenticator) ListKeys(ctx context.Context, tenant uuid.UUID) ([]*Key, error) {
	rows, err := a.db.QueryContext(ctx, `SELECT id, tenant_id, name, prefix, role, created_at, last_used_at, revoked_at
		FROM api_keys WHERE tenant_id = ? ORDER BY created_at DESC`, tenant.String())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Key{}
	for rows.Next() {
		var k Key
		var id, tid string
		var used, revoked sql.NullTime
		if err := rows.Scan(&id, &tid, &k.Name, &k.Prefix, &k.Role, &k.CreatedAt, &used, &revoked); err != nil {
			return nil, err
		}
		k.ID, k.TenantID = uuid.MustParse(id), uuid.MustParse(tid)
		if used.Valid {
			k.LastUsedAt = &used.Time
		}
		if revoked.Valid {
			k.RevokedAt = &revoked.Time
		}
		out = append(out, &k)
	}
	return out, rows.Err()
}

// RevokeKey revokes a key of the given tenant.
func (a *Authenticator) RevokeKey(ctx context.Context, tenant, id uuid.UUID) error {
	res, err := a.db.ExecContext(ctx, `UPDATE api_keys SET revoked_at = NOW(6)
		WHERE id = ? AND tenant_id = ? AND revoked_at IS NULL`, id.String(), tenant.String())
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	a.Forget()
	return nil
}

// BearerToken extracts the token from an Authorization header. ok is false
// when the header is missing or not of the form "Bearer <token>".
func BearerToken(header string) (string, bool) {
	t, ok := strings.CutPrefix(header, "Bearer ")
	t = strings.TrimSpace(t)
	return t, ok && t != ""
}
