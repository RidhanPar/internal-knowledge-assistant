"""Data-access layer: all SQL lives here.

Keeping queries in one module means the rest of the app talks to storage through
named methods, not ad-hoc SQL — easier to test, and the vector search is in one
auditable place.
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from app.ingestion.chunker import Chunk


@dataclass
class RetrievedChunk:
    document_id: str
    document_title: str
    source_path: str
    heading_path: str | None
    content: str
    similarity: float  # cosine similarity in 0..1 (1 - distance)


class Repository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # --- Ingestion --------------------------------------------------------
    async def get_document_hash(self, document_id: str) -> str | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT content_hash FROM documents WHERE id = $1", document_id
            )

    async def upsert_document_with_chunks(
        self,
        *,
        document_id: str,
        title: str,
        source_path: str,
        content_hash: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Replace a document and its chunks atomically.

        Re-ingestion is delete-then-insert within a transaction: the chunk rows
        cascade-delete with the document, so a re-run can never leave a document
        half-updated or mix stale and fresh chunks.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
                await conn.execute(
                    """
                    INSERT INTO documents (id, title, source_path, content_hash)
                    VALUES ($1, $2, $3, $4)
                    """,
                    document_id,
                    title,
                    source_path,
                    content_hash,
                )
                await conn.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, heading_path, content, token_count, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    [
                        (
                            document_id,
                            c.chunk_index,
                            c.heading_path or None,
                            c.content,
                            c.token_count,
                            emb,
                        )
                        for c, emb in zip(chunks, embeddings)
                    ],
                )

    # --- Retrieval --------------------------------------------------------
    async def search(
        self, embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks by cosine similarity to `embedding`.

        `<=>` is pgvector's cosine-distance operator and uses the HNSW index.
        We compute similarity = 1 - distance in SQL so callers get an intuitive
        0..1 relevance score without touching raw distances.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.document_id,
                    d.title        AS document_title,
                    d.source_path  AS source_path,
                    c.heading_path,
                    c.content,
                    1 - (c.embedding <=> $1) AS similarity
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> $1
                LIMIT $2
                """,
                embedding,
                top_k,
            )
        return [
            RetrievedChunk(
                document_id=r["document_id"],
                document_title=r["document_title"],
                source_path=r["source_path"],
                heading_path=r["heading_path"],
                content=r["content"],
                similarity=float(r["similarity"]),
            )
            for r in rows
        ]

    # --- Ops / health -----------------------------------------------------
    async def counts(self) -> tuple[int, int]:
        async with self._pool.acquire() as conn:
            docs = await conn.fetchval("SELECT count(*) FROM documents")
            chunks = await conn.fetchval("SELECT count(*) FROM chunks")
        return int(docs), int(chunks)

    async def ping(self) -> bool:
        async with self._pool.acquire() as conn:
            return (await conn.fetchval("SELECT 1")) == 1
