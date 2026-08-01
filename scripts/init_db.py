"""Create the schema on a fresh database.

The local docker-compose database runs db/init/001_schema.sql automatically on
first boot. A managed database like RDS does not, so this script applies the same
SQL over a direct connection.

It connects with a plain asyncpg connection rather than the application pool. The
pool registers the pgvector type on every connection, which only works after the
`vector` extension exists, and this script is what creates that extension. Using
a plain connection avoids that ordering problem.

    py -3.11 -m scripts.init_db
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger("init_db")
SCHEMA_FILE = Path("db/init/001_schema.sql")


async def _run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    conn = await asyncpg.connect(settings.database_url)
    try:
        # No parameters, so asyncpg uses the simple query protocol and runs all
        # statements in the file in one call.
        await conn.execute(sql)
        logger.info("schema applied from %s", SCHEMA_FILE)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(_run())
