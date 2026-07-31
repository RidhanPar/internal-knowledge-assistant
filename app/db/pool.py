"""asyncpg connection pool with pgvector type registration.

We use asyncpg directly rather than an ORM: the interesting query here is a
vector similarity search, and writing it as explicit SQL keeps the pgvector
operators (`<=>`) visible and defensible. The pool is created once at app
startup and shared across requests.
"""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Teach this connection how to encode/decode the `vector` type so we can
    # pass Python lists/np arrays as parameters and read them back.
    await register_vector(conn)


async def create_pool(settings: Settings) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        init=_init_connection,
        command_timeout=30,
    )
