"""Query endpoints.

- POST /query        — the agentic path (LangGraph agent, multi-tool, multi-step).
- POST /query/simple — the Phase 1 single-shot RAG baseline, kept for comparison.

Both return the same `QueryResponse` shape. The agent path additionally populates
`steps` with the tools it invoked.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agent.service import AgentService
from app.api.deps import get_agent_service, get_rag_service
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.service import RagService

router = APIRouter(tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    service: AgentService = Depends(get_agent_service),
) -> QueryResponse:
    """Answer a question with the agent: it decides whether to retrieve, can run
    multiple tools over multiple steps, and returns `no_answer=true` when its
    tools don't support an answer rather than hallucinating.
    """
    return await service.answer(payload.question, top_k=payload.top_k)


@router.post("/query/simple", response_model=QueryResponse)
async def query_simple(
    payload: QueryRequest,
    service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    """Single-shot RAG baseline: retrieve once, then answer. No agentic loop."""
    return await service.answer(payload.question, top_k=payload.top_k)
