"""The grounding prompt is a correctness surface, so its shape is tested."""

from __future__ import annotations

from app.db.repository import RetrievedChunk
from app.rag import prompts


def _chunk(title: str, heading: str, content: str, sim: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=title.lower(),
        document_title=title,
        source_path=f"corpus/{title.lower()}.md",
        heading_path=heading,
        content=content,
        similarity=sim,
    )


def test_sources_block_is_1_indexed_and_labelled():
    evidence = [
        _chunk("Security Policy", "Access Control > MFA", "MFA is mandatory."),
        _chunk("Handbook", "Annual Leave", "28 days of leave."),
    ]
    block = prompts.build_sources_block(evidence)
    assert "[1]" in block and "[2]" in block
    assert "Security Policy" in block
    assert "Access Control > MFA" in block


def test_system_prompt_enforces_grounding_and_sentinel():
    assert prompts.NO_ANSWER_SENTINEL in prompts.SYSTEM_PROMPT
    lowered = prompts.SYSTEM_PROMPT.lower()
    assert "only" in lowered  # answer only from sources
    assert "cite" in lowered


def test_is_refusal_accepts_bare_and_grounded_refusals():
    # Bare sentinel.
    assert prompts.is_refusal(prompts.NO_ANSWER_SENTINEL) is True
    # Grounded refusal: explains what the docs cover, then ends with the sentinel.
    grounded = (
        "The documents describe medical insurance but do not mention a dental "
        "provider [1]. " + prompts.NO_ANSWER_SENTINEL
    )
    assert prompts.is_refusal(grounded) is True


def test_is_refusal_rejects_a_real_answer():
    assert prompts.is_refusal("Employees get 28 days of leave [1].") is False
