"""Health / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_repository
from app.db.repository import Repository
from app.models.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(repo: Repository = Depends(get_repository)) -> HealthResponse:
    db_ok = False
    docs = chunks = 0
    try:
        db_ok = await repo.ping()
        docs, chunks = await repo.counts()
    except Exception:  # noqa: BLE001 - health must never raise
        db_ok = False
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=__version__,
        db="up" if db_ok else "down",
        corpus_documents=docs,
        corpus_chunks=chunks,
    )
