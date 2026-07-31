"""RagService control-flow tests with fakes — no Bedrock, no database.

These lock in the behaviours that make the assistant trustworthy: refusing when
there is no evidence, and deriving citations from the markers the model actually
used.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.db.repository import RetrievedChunk
from app.rag import prompts, service as service_module
from app.rag.retriever import RetrievalOutcome
from app.rag.llm import LLMResult
from app.rag.service import RagService


class _FakeRetriever:
    def __init__(self, outcome: RetrievalOutcome) -> None:
        self._outcome = outcome

    async def retrieve(self, question: str, top_k=None) -> RetrievalOutcome:
        return self._outcome


def _evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk("security-policy", "Security Policy", "corpus/security-policy.md",
                       "Authentication > MFA", "MFA is mandatory for all accounts.", 0.71),
        RetrievedChunk("employee-handbook", "Employee Handbook", "corpus/employee-handbook.md",
                       "Annual Leave", "Employees accrue 28 days of leave.", 0.55),
    ]


def _service(outcome: RetrievalOutcome) -> RagService:
    return RagService(_FakeRetriever(outcome), Settings())


async def test_no_evidence_short_circuits_to_no_answer(monkeypatch):
    called = False

    async def _should_not_run(*a, **k):
        nonlocal called
        called = True
        return LLMResult("nope", 0, 0, "end_turn")

    monkeypatch.setattr(service_module, "generate", _should_not_run)
    svc = _service(RetrievalOutcome(retrieved=[], evidence=[]))

    resp = await svc.answer("What is the capital of France?")
    assert resp.no_answer is True
    assert resp.answer == prompts.NO_ANSWER_SENTINEL
    assert resp.citations == []
    assert called is False  # the generator must not be invoked without evidence


async def test_citations_reflect_markers_used_in_answer(monkeypatch):
    async def _fake_generate(system, user):
        return LLMResult("MFA is mandatory for all accounts [1].", 100, 20, "end_turn")

    monkeypatch.setattr(service_module, "generate", _fake_generate)
    ev = _evidence()
    svc = _service(RetrievalOutcome(retrieved=ev, evidence=ev))

    resp = await svc.answer("Is MFA required?")
    assert resp.no_answer is False
    # Only [1] was cited, so only that source is returned.
    assert [c.marker for c in resp.citations] == [1]
    assert resp.citations[0].document_id == "security-policy"


async def test_model_no_answer_sentinel_sets_flag(monkeypatch):
    async def _fake_generate(system, user):
        return LLMResult(prompts.NO_ANSWER_SENTINEL, 100, 8, "end_turn")

    monkeypatch.setattr(service_module, "generate", _fake_generate)
    ev = _evidence()
    svc = _service(RetrievalOutcome(retrieved=ev, evidence=ev))

    resp = await svc.answer("Something unanswerable from context")
    assert resp.no_answer is True
    assert resp.citations == []


async def test_uncited_answer_falls_back_to_all_evidence(monkeypatch):
    async def _fake_generate(system, user):
        return LLMResult("Answer with no bracket markers at all.", 100, 12, "end_turn")

    monkeypatch.setattr(service_module, "generate", _fake_generate)
    ev = _evidence()
    svc = _service(RetrievalOutcome(retrieved=ev, evidence=ev))

    resp = await svc.answer("A question")
    assert [c.marker for c in resp.citations] == [1, 2]


@pytest.mark.parametrize("answer,expected", [
    ("uses [1] and [2] and [1] again", [1, 2]),
    ("out of range [9] ignored, [1] kept", [1]),
    ("no markers", []),
])
def test_marker_extraction(answer, expected):
    assert service_module._cited_markers(answer, n_sources=2) == expected
