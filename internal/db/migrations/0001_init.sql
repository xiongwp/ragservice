-- MySQL 8.0.16+ (CHECK constraints, expression defaults, SKIP LOCKED).
-- Vectors live in Qdrant; MySQL is the source of truth for everything else,
-- and a vector hit only counts if its chunk row exists here.

CREATE TABLE IF NOT EXISTS datasets (
  id              CHAR(36) CHARACTER SET ascii NOT NULL PRIMARY KEY,
  tenant_id       CHAR(36) CHARACTER SET ascii NOT NULL,
  name            VARCHAR(255) NOT NULL,
  description     TEXT NOT NULL,
  embedding_model VARCHAR(191) NOT NULL,
  chunk_config    JSON NOT NULL,  -- {"parent_tokens": 800, "child_tokens": 200}
  created_at      DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at      DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  KEY datasets_tenant_idx (tenant_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS documents (
  id           CHAR(36) CHARACTER SET ascii NOT NULL PRIMARY KEY,
  tenant_id    CHAR(36) CHARACTER SET ascii NOT NULL,
  dataset_id   CHAR(36) CHARACTER SET ascii NOT NULL,
  name         VARCHAR(512) NOT NULL,
  format       VARCHAR(16) NOT NULL,   -- txt | md | html | pdf | docx | pptx | xlsx
  size_bytes   BIGINT NOT NULL,
  content_hash BINARY(32) NOT NULL,    -- sha256 of the uploaded bytes
  metadata     JSON NOT NULL,
  status       VARCHAR(16) NOT NULL DEFAULT 'queued',
  stage        VARCHAR(16) NOT NULL DEFAULT '',  -- parsing | chunking | embedding | indexing
  error        TEXT NOT NULL DEFAULT (''),
  chunk_count  INT NOT NULL DEFAULT 0,
  token_count  INT NOT NULL DEFAULT 0,
  created_at   DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at   DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  indexed_at   DATETIME(6) NULL,
  UNIQUE KEY documents_dataset_hash (dataset_id, content_hash),
  KEY documents_dataset_idx (dataset_id, created_at),
  CONSTRAINT documents_status_chk CHECK (status IN ('queued', 'processing', 'retrying', 'completed', 'failed')),
  CONSTRAINT documents_dataset_fk FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Raw upload bytes, kept apart so listing documents never drags blobs along.
CREATE TABLE IF NOT EXISTS document_blobs (
  document_id CHAR(36) CHARACTER SET ascii NOT NULL PRIMARY KEY,
  data        LONGBLOB NOT NULL,
  CONSTRAINT document_blobs_fk FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Parents carry the context returned to the LLM; children are what gets
-- embedded (one Qdrant point per child, same id). parent_id has no foreign
-- key on purpose: InnoDB treats a cascade that re-enters the same table as
-- RESTRICT, and chunks are only ever replaced a whole document at a time.
CREATE TABLE IF NOT EXISTS chunks (
  id           CHAR(36) CHARACTER SET ascii NOT NULL PRIMARY KEY,
  tenant_id    CHAR(36) CHARACTER SET ascii NOT NULL,
  dataset_id   CHAR(36) CHARACTER SET ascii NOT NULL,
  document_id  CHAR(36) CHARACTER SET ascii NOT NULL,
  parent_id    CHAR(36) CHARACTER SET ascii NULL,
  kind         VARCHAR(8) NOT NULL,
  position     INT NOT NULL,
  heading      TEXT NOT NULL,         -- "H1 > H2" path where the chunk starts
  content      MEDIUMTEXT NOT NULL,
  tokens       INT NOT NULL,
  content_hash BINARY(32) NOT NULL,   -- sha256 of the embedding input (children) or content (parents)
  enabled      BOOL NOT NULL DEFAULT TRUE,
  KEY chunks_document_idx (document_id, kind, position),
  CONSTRAINT chunks_kind_chk CHECK (kind IN ('parent', 'child')),
  CONSTRAINT chunks_document_fk FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS embedding_cache (
  model      VARCHAR(191) NOT NULL,
  hash       BINARY(32) NOT NULL,     -- sha256 of the exact text sent to the model
  embedding  MEDIUMBLOB NOT NULL,     -- little-endian float32
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (model, hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

-- Job queue. live_key enforces "at most one queued-or-running job per
-- unique_key" while letting finished jobs pile up under the same key.
CREATE TABLE IF NOT EXISTS jobs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  kind         VARCHAR(64) NOT NULL,
  args         JSON NOT NULL,
  unique_key   VARCHAR(191) NULL,
  state        VARCHAR(16) NOT NULL DEFAULT 'available',
  attempt      INT NOT NULL DEFAULT 0,
  max_attempts INT NOT NULL,
  run_at       DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  locked_until DATETIME(6) NULL,
  last_error   TEXT NULL,
  created_at   DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at   DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  live_key     VARCHAR(191) GENERATED ALWAYS AS (IF(state IN ('available', 'running'), unique_key, NULL)) STORED,
  UNIQUE KEY jobs_live_key (live_key),
  KEY jobs_fetch_idx (state, run_at),
  KEY jobs_cleanup_idx (state, updated_at),
  CONSTRAINT jobs_state_chk CHECK (state IN ('available', 'running', 'completed', 'failed', 'cancelled'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
