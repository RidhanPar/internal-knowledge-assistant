"""Application configuration.

Settings are loaded once (see `get_settings`) from environment variables and an
optional `.env` file. Using pydantic-settings gives us typed, validated config
with a single source of truth — no scattered `os.getenv` calls.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Postgres / pgvector ---
    database_url: str = Field(
        default="postgresql://ika:ika@localhost:5434/ika",
        description="asyncpg-compatible DSN.",
    )
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10

    # --- AWS Bedrock ---
    aws_region: str = "us-east-1"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 1024
    bedrock_llm_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Bounded concurrency for embedding calls during ingestion. Titan embeds one
    # input per request; this caps in-flight Bedrock calls to stay under account
    # throttling limits.
    embedding_concurrency: int = 8

    # --- Retrieval / generation ---
    retrieval_top_k: int = 6
    retrieval_min_similarity: float = 0.30
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- Agent ---
    # Upper bound on tool-call rounds before the agent must answer. Prevents an
    # unbounded tool loop; generous enough for multi-part questions.
    max_agent_steps: int = 5

    # --- App ---
    log_level: str = "INFO"
    corpus_dir: str = "corpus"
    directory_path: str = "data/directory.json"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
