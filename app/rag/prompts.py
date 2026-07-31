"""Prompt construction for grounded generation.

Isolated from the LLM transport so the exact wording — the thing that most
affects faithfulness — is unit-testable and reviewable in one place.
"""

from __future__ import annotations

from app.db.repository import RetrievedChunk

# Sentinel the model is told to emit verbatim when the context can't support an
# answer. We detect it downstream to set `no_answer`.
NO_ANSWER_SENTINEL = "I don't know based on the available documents."

SYSTEM_PROMPT = f"""You are an internal knowledge assistant for a company. \
You answer employees' questions using ONLY the numbered source excerpts provided \
in each request.

Rules you must follow:
1. Ground every claim in the provided sources. Do not use outside or prior \
knowledge, even if you are confident.
2. Cite the sources you used inline with bracketed numbers matching the source \
list, e.g. "Employees accrue 25 days of leave [2]." Cite every sentence that \
states a fact from the sources.
3. If the sources do not contain enough information to answer, reply with \
exactly this sentence and nothing else: "{NO_ANSWER_SENTINEL}"
4. Do not speculate, hedge with invented details, or pad the answer. Be concise \
and direct.
5. If sources conflict, say so and cite each side."""


def build_sources_block(evidence: list[RetrievedChunk]) -> str:
    """Render evidence as a numbered list the model cites by [n] (1-indexed)."""
    lines: list[str] = []
    for i, chunk in enumerate(evidence, start=1):
        location = chunk.heading_path or chunk.document_title
        lines.append(
            f"[{i}] (source: {chunk.document_title} — {location})\n{chunk.content}"
        )
    return "\n\n".join(lines)


def build_user_prompt(question: str, evidence: list[RetrievedChunk]) -> str:
    sources_block = build_sources_block(evidence)
    return (
        "Sources:\n"
        f"{sources_block}\n\n"
        "---\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, with inline [n] citations."
    )
