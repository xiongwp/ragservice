-- Contextual Retrieval: an LLM-written sentence situating each child chunk
-- in its document, prepended to the text that is embedded and BM25-indexed.
ALTER TABLE chunks ADD COLUMN context TEXT NULL AFTER content;

-- Generated contexts, keyed by what the prompt depends on, so re-indexing
-- an unchanged chunk never calls the LLM again.
CREATE TABLE IF NOT EXISTS context_cache (
  model      VARCHAR(191) NOT NULL,
  hash       BINARY(32) NOT NULL,
  context    TEXT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (model, hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
