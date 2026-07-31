"""Verify Bedrock access before running ingestion or the API.

Makes one small embedding call and one small Converse call using the model IDs
from your .env, and reports dimensions/latency. Run this first to confirm your
AWS credentials, region, and — importantly — that you have *enabled model
access* for both models in the Bedrock console.

    py -3.11 -m scripts.check_bedrock
"""

from __future__ import annotations

import asyncio
import time

from app.config import get_settings
from app.rag.embeddings import embed_query
from app.rag.llm import generate


async def _run() -> None:
    s = get_settings()
    print(f"region={s.aws_region}")
    print(f"embedding_model={s.bedrock_embedding_model_id}")
    print(f"llm_model={s.bedrock_llm_model_id}\n")

    t0 = time.perf_counter()
    vec = await embed_query("hello world")
    print(f"[embeddings] OK dim={len(vec)} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    if len(vec) != s.embedding_dim:
        print(
            f"  WARNING: returned dim {len(vec)} != EMBEDDING_DIM {s.embedding_dim}. "
            "Update EMBEDDING_DIM and the vector(...) column to match."
        )

    t0 = time.perf_counter()
    result = await generate(
        "You are a test. Reply with exactly the word: pong.",
        "ping",
    )
    print(
        f"[generation] OK ({(time.perf_counter()-t0)*1000:.0f} ms) "
        f"in={result.input_tokens} out={result.output_tokens} "
        f"stop={result.stop_reason}\n  reply: {result.text!r}"
    )
    print("\nBedrock access looks good.")


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface the raw error to the operator
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print(
            "\nCommon causes:\n"
            "  - Model access not enabled in the Bedrock console for this region\n"
            "  - Wrong BEDROCK_LLM_MODEL_ID (try `aws bedrock list-inference-profiles`)\n"
            "  - Credentials/region not set (check AWS_REGION / AWS_PROFILE)\n"
        )
        raise SystemExit(1)
