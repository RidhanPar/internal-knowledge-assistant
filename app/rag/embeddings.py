"""Embeddings via a local sentence-transformers model (bge-base-en-v1.5).

Why a local model:
- No extra vendor or API key, and the document text never leaves the server,
  which is the right default for internal documents.
- Free to run and reproducible: the same model file gives the same vectors.
- bge-base-en-v1.5 is a strong open embedding model at 768 dimensions, small
  enough to run on CPU.

Cost of the choice: it pulls in PyTorch and needs about 1 GB of memory, so the
host must be sized for it. The call sits behind `embed_query` and `embed_texts`,
so swapping to a hosted embedding model later is a change to this one file plus a
re-index.

Asymmetric search detail: bge models retrieve better when the query carries a
short instruction prefix and the stored passages do not. We add that prefix to
queries only, which matches how the model was trained.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

# Recommended query instruction for bge-*-en-v1.5. Applied to queries only.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> SentenceTransformer:
    """Load the model once per process. The first call downloads and caches it."""
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def _encode(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings=True returns unit vectors, so cosine distance in
    # pgvector maps to similarity = 1 - distance.
    vectors = _model().encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]


async def embed_query(text: str) -> list[float]:
    """Embed one query. Encoding is CPU-bound, so run it off the event loop."""
    vectors = await asyncio.to_thread(_encode, [_QUERY_PREFIX + text])
    return vectors[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many passages (no query prefix), preserving input order."""
    return await asyncio.to_thread(_encode, texts)
