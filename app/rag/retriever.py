"""Retrieval: embed the question, fetch nearest chunks, apply a relevance gate.

The similarity threshold is what lets the assistant say "I don't know" instead
of confabulating: if nothing clears the bar, there is no evidence to ground an
answer, and we never call the generator on empty context.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.db.repository import Repository, RetrievedChunk
from app.rag.embeddings import embed_query


@dataclass
class RetrievalOutcome:
    # All chunks returned by the vector search (for observability/debugging).
    retrieved: list[RetrievedChunk]
    # The subset that cleared the similarity threshold — the evidence we ground on.
    evidence: list[RetrievedChunk]


class Retriever:
    def __init__(self, repo: Repository, settings: Settings) -> None:
        self._repo = repo
        self._settings = settings

    async def retrieve(self, question: str, top_k: int | None = None) -> RetrievalOutcome:
        k = top_k or self._settings.retrieval_top_k
        query_vec = await embed_query(question)
        retrieved = await self._repo.search(query_vec, k)
        threshold = self._settings.retrieval_min_similarity
        evidence = [c for c in retrieved if c.similarity >= threshold]
        return RetrievalOutcome(retrieved=retrieved, evidence=evidence)
