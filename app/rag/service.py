"""RAG orchestration: retrieve -> ground -> generate -> attach citations.

This is the seam the API depends on. It owns the control flow that makes the
assistant *grounded*: no evidence means no generation call and an explicit
no-answer; citations in the response are derived from the markers the model
actually used, not merely from what was retrieved.
"""

from __future__ import annotations

import time

from app.config import Settings
from app.models.schemas import Citation, QueryResponse
from app.rag import prompts
from app.rag.citations import cited_markers as _cited_markers
from app.rag.llm import generate
from app.rag.retriever import Retriever

_SNIPPET_MAX_CHARS = 600


class RagService:
    def __init__(self, retriever: Retriever, settings: Settings) -> None:
        self._retriever = retriever
        self._settings = settings

    async def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        started = time.perf_counter()
        outcome = await self._retriever.retrieve(question, top_k=top_k)

        # No evidence cleared the relevance gate — refuse rather than guess.
        if not outcome.evidence:
            return QueryResponse(
                question=question,
                answer=prompts.NO_ANSWER_SENTINEL,
                no_answer=True,
                citations=[],
                model_id=self._settings.bedrock_llm_model_id,
                retrieved_count=len(outcome.retrieved),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        evidence = outcome.evidence
        user_prompt = prompts.build_user_prompt(question, evidence)
        result = await generate(prompts.SYSTEM_PROMPT, user_prompt)

        no_answer = result.text.strip() == prompts.NO_ANSWER_SENTINEL
        markers = [] if no_answer else _cited_markers(result.text, len(evidence))

        # Map each used marker back to its source chunk. If the model answered
        # but cited nothing (rare, against instructions), fall back to exposing
        # all evidence so the answer is never uncitable.
        cited_indices = markers or ([] if no_answer else list(range(1, len(evidence) + 1)))
        citations = [
            Citation(
                marker=i,
                document_id=evidence[i - 1].document_id,
                document_title=evidence[i - 1].document_title,
                heading_path=evidence[i - 1].heading_path,
                source_path=evidence[i - 1].source_path,
                similarity=round(evidence[i - 1].similarity, 4),
                snippet=evidence[i - 1].content[:_SNIPPET_MAX_CHARS],
            )
            for i in cited_indices
        ]

        return QueryResponse(
            question=question,
            answer=result.text,
            no_answer=no_answer,
            citations=citations,
            model_id=self._settings.bedrock_llm_model_id,
            retrieved_count=len(outcome.retrieved),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
