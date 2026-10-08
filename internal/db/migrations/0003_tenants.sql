-- Tenants, their limits, and API keys. Limits of 0 mean unlimited.
CREATE TABLE IF NOT EXISTS tenants (
  id             CHAR(36) CHARACTER SET ascii NOT NULL PRIMARY KEY,
  name           VARCHAR(255) NOT NULL,
  rate_limit_rps DOUBLE NOT NULL DEFAULT 0,   -- requests per second across all of the tenant's keys
  max_datasets   INT NOT NULL DEFAULT 0,
  max_documents  INT NOT NULL DEFAULT 0,
  max_storage_mb INT NOT NULL DEFAULT 0,      -- total size of uploaded files
  created_at     DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at     DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Only a hash of each key is stored; the key itself is shown once, at creation.
CREATE TABLE IF NOT EXISTS api_keys (
  id           CHAR(36) CHARACTER SET ascii NOT NULL PRIMARY KEY,
  tenant_id    CHAR(36) CHARACTER SET ascii NOT NULL,
  name         VARCHAR(255) NOT NULL,
  prefix       VARCHAR(16) NOT NULL,        -- first characters, to tell keys apart in listings
  key_hash     BINARY(32) NOT NULL,         -- sha256 of the full key
  role         VARCHAR(16) NOT NULL,
  created_at   DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  last_used_at DATETIME(6) NULL,
  revoked_at   DATETIME(6) NULL,
  UNIQUE KEY api_keys_hash (key_hash),
  KEY api_keys_tenant_idx (tenant_id, created_at),
  CONSTRAINT api_keys_role_chk CHECK (role IN ('admin', 'editor', 'reader')),
  CONSTRAINT api_keys_tenant_fk FOREIGN KEY (tenant_id) REFERENCES tenants (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
