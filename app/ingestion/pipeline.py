"""Ingestion pipeline: load -> chunk -> embed -> store.

Idempotent by content hash: a document whose file is unchanged since the last
run is skipped, so re-ingesting the corpus is cheap and safe to run repeatedly
(e.g. from a deploy hook). Changed documents are fully replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.db.repository import Repository
from app.ingestion.chunker import chunk_markdown
from app.ingestion.loader import load_corpus
from app.rag.embeddings import embed_texts

logger = get_logger(__name__)


@dataclass
class IngestReport:
    documents_processed: int
    documents_skipped: int
    chunks_written: int


async def ingest_corpus(
    repo: Repository, corpus_dir: str, *, force: bool = False
) -> IngestReport:
    docs = load_corpus(corpus_dir)
    processed = skipped = chunks_written = 0

    for doc in docs:
        if not force:
            existing = await repo.get_document_hash(doc.id)
            if existing == doc.content_hash:
                logger.info("skip unchanged: %s", doc.id)
                skipped += 1
                continue

        chunks = chunk_markdown(doc.id, doc.content)
        if not chunks:
            logger.warning("no chunks produced for %s; skipping", doc.id)
            continue

        # Embed the chunk text exactly as stored (heading path already prepended
        # by the chunker) so query and document embeddings share the same space.
        embeddings = await embed_texts([c.content for c in chunks])

        await repo.upsert_document_with_chunks(
            document_id=doc.id,
            title=doc.title,
            source_path=doc.source_path,
            content_hash=doc.content_hash,
            chunks=chunks,
            embeddings=embeddings,
        )
        logger.info("ingested %s: %d chunks", doc.id, len(chunks))
        processed += 1
        chunks_written += len(chunks)

    return IngestReport(
        documents_processed=processed,
        documents_skipped=skipped,
        chunks_written=chunks_written,
    )
