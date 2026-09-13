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

    # --- Embeddings (local sentence-transformers) ---
    # bge-base-en-v1.5 emits 768-dim vectors. This value MUST match the vector
    # column size in db/init/001_schema.sql. Changing the model means a re-index.
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Generation (Anthropic API) ---
    # The SDK also reads ANTHROPIC_API_KEY from the environment; this field lets
    # it come from a .env file too. Never hardcode the key.
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    # Only needed if the API key is organization-scoped rather than
    # workspace-scoped. Leave blank when using a workspace-scoped key.
    anthropic_workspace_id: str = ""

    # --- Retrieval / generation ---
    retrieval_top_k: int = 6
    # Minimum cosine similarity for a chunk to count as evidence. bge-base scores
    # real matches around 0.7 and unrelated text around 0.3, so 0.40 cleanly
    # separates them. Tuned against the evaluation set in Phase 3.
    retrieval_min_similarity: float = 0.40
    llm_max_tokens: int = 1024

    # --- Agent ---
    # Upper bound on tool-call rounds before the agent must answer. Prevents an
    # unbounded tool loop; generous enough for multi-part questions.
    max_agent_steps: int = 5

    # --- App ---
    log_level: str = "INFO"
    # Emit one JSON object per log line. Turn on in deployment so a log
    # aggregator can index the fields; leave off locally for readable text.
    json_logs: bool = False
    corpus_dir: str = "corpus"
    directory_path: str = "data/directory.json"
    # Port the API listens on. App Runner and most container platforms set PORT.
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
