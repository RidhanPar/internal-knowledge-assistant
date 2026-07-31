"""Embedding via Amazon Titan Text Embeddings v2 on Bedrock.

Why Titan v2:
- Native to Bedrock, so no extra vendor / key to manage alongside Claude.
- Configurable output dimension (256/512/1024). We use 1024 for headroom on a
  heterogeneous internal corpus; smaller dims would cut storage/latency if the
  corpus were narrow.
- Returns L2-normalised vectors (`normalize=true`), so cosine distance in
  pgvector is directly interpretable as `similarity = 1 - distance`.

Titan embeds a single input per request, so batch embedding is fan-out with
bounded concurrency to respect account throttling limits.
"""

from __future__ import annotations

import asyncio
import json

from app.config import get_settings
from app.rag.bedrock import get_bedrock_runtime


def _embed_sync(text: str) -> list[float]:
    settings = get_settings()
    client = get_bedrock_runtime()
    body = json.dumps(
        {
            "inputText": text,
            "dimensions": settings.embedding_dim,
            "normalize": True,
        }
    )
    resp = client.invoke_model(
        modelId=settings.bedrock_embedding_model_id,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload["embedding"]


async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return await asyncio.to_thread(_embed_sync, text)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts with bounded concurrency, preserving input order."""
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.embedding_concurrency)

    async def _one(t: str) -> list[float]:
        async with semaphore:
            return await asyncio.to_thread(_embed_sync, t)

    return await asyncio.gather(*(_one(t) for t in texts))
