"""Verify the models are reachable before ingesting or running the API.

Loads the local embedding model and makes one small Anthropic call, then reports
the embedding dimension and the reply. Run this first to confirm the embedding
model downloads and that ANTHROPIC_API_KEY works.

    py -3.11 -m scripts.check_models
"""

from __future__ import annotations

import asyncio
import time

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.rag.llm import generate


async def _run() -> None:
    s = get_settings()
    print(f"embedding_model={s.embedding_model_name}")
    print(f"anthropic_model={s.anthropic_model}\n")

    t0 = time.perf_counter()
    vec = await embed_query("hello world")
    print(f"[embeddings] OK dim={len(vec)} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    if len(vec) != s.embedding_dim:
        print(
            f"  WARNING: returned dim {len(vec)} != EMBEDDING_DIM {s.embedding_dim}. "
            "Update EMBEDDING_DIM and the vector(...) column to match."
        )

    t0 = time.perf_counter()
    result = await generate("You are a test. Reply with exactly the word: pong.", "ping")
    print(
        f"[generation] OK ({(time.perf_counter()-t0)*1000:.0f} ms) "
        f"in={result.input_tokens} out={result.output_tokens} "
        f"stop={result.stop_reason}\n  reply: {result.text!r}"
    )
    print("\nModels look good.")


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface the raw error to the operator
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print(
            "\nCommon causes:\n"
            "  - ANTHROPIC_API_KEY not set (check your .env or environment)\n"
            "  - Wrong ANTHROPIC_MODEL id\n"
            "  - First run still downloading the embedding model (retry once)\n"
        )
        raise SystemExit(1)
