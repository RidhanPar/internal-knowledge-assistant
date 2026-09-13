-- ─────────────────────────────────────────────────────────────────────────────
-- Schema for the internal knowledge assistant.
--
-- Runs once on first boot of a fresh Postgres volume (docker-entrypoint-initdb).
-- To re-run after edits: `docker compose down -v` (drops the volume) then `up`.
--
-- IMPORTANT: the vector(768) dimension below must match EMBEDDING_DIM in .env.
-- bge-base-en-v1.5 emits 768 dims. Switching embedding model or dimension
-- requires altering this column and rebuilding the HNSW index. There is no
-- automatic migration.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document in the corpus.
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,              -- slug derived from filename
    title        TEXT        NOT NULL,
    source_path  TEXT        NOT NULL,
    content_hash TEXT        NOT NULL,          -- sha256 of raw file; enables skip-on-unchanged
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per retrievable chunk. Chunks belong to a document and cascade-delete
-- with it, which keeps re-ingestion simple: delete the doc's chunks, re-insert.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT       NOT NULL,            -- ordinal within the document
    heading_path TEXT,                          -- e.g. "Security Policy > Access Control > MFA"
    content      TEXT      NOT NULL,
    token_count  INT,
    embedding    vector(768) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- Approximate-nearest-neighbour index for cosine distance (<=>). HNSW gives
-- better recall/latency than IVFFlat at this corpus size and needs no training
-- step. Embeddings are L2-normalised at write time, so cosine ranks identically
-- to inner product while keeping similarity interpretable as 1 - distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
