"""CLI entrypoint to (re-)ingest the corpus.

Usage (from project root, with .env populated and Postgres up):

    py -3.11 -m scripts.ingest            # ingest changed docs only
    py -3.11 -m scripts.ingest --force    # re-embed and replace everything

Runs the same pipeline the API would, against the same database, so the
retrieval endpoint is immediately queryable afterwards.
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.pool import create_pool
from app.db.repository import Repository
from app.ingestion.pipeline import ingest_corpus

logger = get_logger("ingest")


async def _run(force: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    pool = await create_pool(settings)
    try:
        repo = Repository(pool)
        report = await ingest_corpus(repo, settings.corpus_dir, force=force)
        logger.info(
            "done: %d processed, %d skipped, %d chunks written",
            report.documents_processed,
            report.documents_skipped,
            report.chunks_written,
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into pgvector.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and replace all documents, ignoring content-hash skip.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.force))


if __name__ == "__main__":
    main()
