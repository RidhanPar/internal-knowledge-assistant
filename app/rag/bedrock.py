"""Bedrock client factory.

boto3 clients are thread-safe for calls but creation is not cheap, so we build
one `bedrock-runtime` client per process and reuse it. boto3 is synchronous;
callers wrap invocations in `asyncio.to_thread` so the event loop is never
blocked (see embeddings.py / llm.py).
"""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.config import Config

from app.config import get_settings


@lru_cache
def get_bedrock_runtime():
    settings = get_settings()
    # Adaptive retries handle Bedrock throttling gracefully during ingestion
    # bursts; a generous read timeout accommodates longer generations.
    config = Config(
        region_name=settings.aws_region,
        retries={"max_attempts": 5, "mode": "adaptive"},
        read_timeout=60,
        connect_timeout=10,
    )
    return boto3.client("bedrock-runtime", config=config)
