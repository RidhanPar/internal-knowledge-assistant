"""API request/response schemas.

These are the public contract of the service. Retrieval internals (raw distances,
embeddings) are deliberately not exposed except through the debug-friendly
`sources` list, which carries enough provenance for a UI to render citations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of chunks retrieved for this query.",
    )


class Citation(BaseModel):
    """A source chunk the answer is grounded in."""

    marker: int = Field(..., description="The [n] marker used inline in the answer.")
    document_id: str
    document_title: str
    heading_path: str | None
    source_path: str
    similarity: float = Field(..., description="Cosine similarity to the query, 0..1.")
    snippet: str = Field(..., description="The retrieved chunk text (possibly truncated).")


class QueryResponse(BaseModel):
    question: str
    answer: str
    # True when retrieval found no sufficiently-relevant evidence, or the model
    # declined to answer from the given context. Callers should surface this
    # rather than treating the answer as authoritative.
    no_answer: bool
    citations: list[Citation]
    model_id: str
    retrieved_count: int
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
    corpus_documents: int
    corpus_chunks: int
