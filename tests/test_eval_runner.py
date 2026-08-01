"""Evaluator aggregation, driven by fakes. No Bedrock, no database.

Proves the runner attributes results to the right buckets: retrieval metrics from
the retriever, behaviour from the agent's answer/refuse decision, and
faithfulness from the judge, including the flagging of hallucinated answers.
"""

from __future__ import annotations

from app.db.repository import RetrievedChunk
from app.eval import runner as runner_module
from app.eval.dataset import EvalCase
from app.eval.faithfulness import FaithfulnessResult
from app.eval.runner import Evaluator
from app.models.schemas import Citation, QueryResponse
from app.rag.retriever import RetrievalOutcome


class FakeRetriever:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        # question -> ordered list of retrieved doc ids
        self._mapping = mapping

    async def retrieve(self, question: str, top_k=None) -> RetrievalOutcome:
        ids = self._mapping.get(question, [])
        chunks = [
            RetrievedChunk(doc_id, doc_id, f"corpus/{doc_id}.md", None, "text", 0.6)
            for doc_id in ids
        ]
        return RetrievalOutcome(retrieved=chunks, evidence=chunks)


class FakeAgent:
    def __init__(self, mapping: dict[str, QueryResponse]) -> None:
        self._mapping = mapping

    async def answer(self, question: str, top_k=None) -> QueryResponse:
        return self._mapping[question]


def _answer(text: str, cited: bool = True) -> QueryResponse:
    citations = (
        [Citation(marker=1, document_id="d", document_title="D", source_path="p", snippet="s")]
        if cited
        else []
    )
    return QueryResponse(
        question="", answer=text, no_answer=False, citations=citations,
        model_id="m", retrieved_count=len(citations), latency_ms=1,
    )


def _refusal() -> QueryResponse:
    return QueryResponse(
        question="", answer="I don't know based on the available documents.",
        no_answer=True, citations=[], model_id="m", retrieved_count=0, latency_ms=1,
    )


async def test_runner_aggregates_all_dimensions(monkeypatch):
    async def fake_judge(question, answer, sources_text):
        verdict = "unsupported" if "MADE UP" in answer else "supported"
        score = 0.0 if verdict == "unsupported" else 1.0
        return FaithfulnessResult(verdict=verdict, score=score)

    monkeypatch.setattr(runner_module, "judge_faithfulness", fake_judge)

    cases = [
        EvalCase("good", "q-good", "policy", False, relevant_doc_ids=["handbook"],
                 answer_must_include=["28"]),
        EvalCase("norefuse", "q-unanswerable", "unanswerable", True),
        EvalCase("halluc", "q-hallucinated", "unanswerable", True),
        EvalCase("ungrounded", "q-ungrounded", "policy", False, relevant_doc_ids=["security-policy"]),
    ]

    retrieval = {
        "q-good": ["handbook", "benefits"],           # relevant first -> RR 1.0
        "q-unanswerable": ["handbook"],
        "q-hallucinated": ["handbook"],
        "q-ungrounded": ["benefits", "security-policy"],  # relevant second -> RR 0.5
    }
    answers = {
        "q-good": _answer("Employees get 28 days [1]."),          # answered, faithful, matches
        "q-unanswerable": _refusal(),                              # refused correctly
        "q-hallucinated": _answer("The price was 42 [1]."),       # false answer -> flagged
        "q-ungrounded": _answer("Something MADE UP [1]."),         # unfaithful -> flagged
    }

    report = await Evaluator(
        FakeRetriever(retrieval), FakeAgent(answers), top_k=6
    ).run(cases)

    # Retrieval scored only over the two policy cases (they have relevant docs).
    assert report.retrieval_scored_count == 2
    assert report.hit_at_k == 1.0
    assert report.mrr == 0.75  # (1.0 + 0.5) / 2

    b = report.behaviour
    assert b.answered_correctly == 2   # good + ungrounded both answered
    assert b.refused_correctly == 1    # unanswerable refused
    assert b.false_answer == 1         # hallucinated answered instead of refusing

    # Faithfulness judged only on answered cases with citations (3 of them).
    assert report.faithfulness_judged_count == 3
    # Two flagged: the false answer, and the ungrounded (unsupported) answer.
    assert report.flagged_count == 2
    # Content match: only the "good" case had answer_must_include and it matched.
    assert report.answer_match_rate == 1.0
