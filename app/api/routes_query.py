"""The core RAG endpoint: question in, grounded cited answer out."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.service import RagService

router = APIRouter(tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer a question over the internal corpus with citations.

    Returns `no_answer=true` (and the sentinel answer) when retrieval finds no
    sufficiently relevant evidence — the caller should treat that as "not in the
    docs", not as a failure.
    """
    return await service.answer(payload.question, top_k=payload.top_k)
