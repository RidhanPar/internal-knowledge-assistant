"""FastAPI application factory and lifespan wiring.

Startup builds the shared resources once — the asyncpg pool, the repository, and
the RAG service — and stashes them on `app.state`. The lifespan context guarantees
the pool is closed cleanly on shutdown. Constructing the app in a factory keeps it
importable by tests without side effects at import time.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.agent.directory import DirectoryService
from app.agent.graph import KnowledgeAgent
from app.agent.service import AgentService
from app.api import routes_health, routes_query
from app.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.pool import create_pool
from app.db.repository import Repository
from app.rag.retriever import Retriever
from app.rag.service import RagService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    logger.info("starting up: connecting to database")
    pool = await create_pool(settings)
    repository = Repository(pool)
    retriever = Retriever(repository, settings)
    directory = DirectoryService.from_json(settings.directory_path)
    agent = KnowledgeAgent(retriever, directory)

    app.state.pool = pool
    app.state.repository = repository
    app.state.rag_service = RagService(retriever, settings)
    app.state.agent_service = AgentService(agent, settings)
    logger.info("startup complete")
    try:
        yield
    finally:
        logger.info("shutting down: closing database pool")
        await pool.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Internal Knowledge Assistant",
        version=__version__,
        summary="Agentic RAG over an internal document corpus with grounded, cited answers.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(routes_health.router)
    app.include_router(routes_query.router)
    return app


app = create_app()
