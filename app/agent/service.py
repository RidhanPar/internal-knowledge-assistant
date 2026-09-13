"""AgentService: run the graph for one question and shape the API response.

This is the seam the `/query` endpoint depends on. It owns the initial state,
invokes the compiled graph, and maps the terminal state onto the same
`QueryResponse` contract the single-shot path uses — so callers get citations,
`no_answer`, and now a `steps` trace, regardless of which path answered.
"""

from __future__ import annotations

import time

from app.agent.graph import KnowledgeAgent
from app.agent.tools import Source
from app.config import Settings
from app.models.schemas import Citation, QueryResponse

_SNIPPET_MAX_CHARS = 600


def _source_to_citation(src: Source) -> Citation:
    return Citation(
        marker=src.marker,
        source_type=src.source_type,
        document_id=src.reference,
        document_title=src.title,
        heading_path=src.location,
        source_path=src.reference,
        similarity=src.similarity,
        snippet=src.snippet[:_SNIPPET_MAX_CHARS],
    )


class AgentService:
    def __init__(self, agent: KnowledgeAgent, settings: Settings) -> None:
        self._agent = agent
        self._settings = settings

    async def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        started = time.perf_counter()
        initial: dict = {
            "question": question,
            "messages": [{"role": "user", "content": question}],
            "sources": [],
            "steps": [],
            "iterations": 0,
            "max_iterations": self._settings.max_agent_steps,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        final = await self._agent.graph.ainvoke(initial)

        sources: list[Source] = final.get("sources", [])
        used_markers: list[int] = final.get("used_markers", [])
        by_marker = {s.marker: s for s in sources}
        citations = [
            _source_to_citation(by_marker[m]) for m in used_markers if m in by_marker
        ]

        return QueryResponse(
            question=question,
            answer=final.get("final_text", ""),
            no_answer=bool(final.get("no_answer", True)),
            citations=citations,
            model_id=self._settings.anthropic_model,
            retrieved_count=len(sources),
            latency_ms=int((time.perf_counter() - started) * 1000),
            steps=final.get("steps", []),
        )
