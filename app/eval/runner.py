"""Run the evaluation dataset against the live system and aggregate the scores.

For each case we do three independent measurements:

1. Retrieval: run the retriever and compare the documents it returns against the
   labelled relevant documents. This is scored only for cases where we labelled
   relevant documents (the policy cases).
2. Behaviour: ask the agent the question and check whether it answered or refused,
   against what the case expects. This is where unanswerable cases catch
   hallucination.
3. Faithfulness: for answered cases, ask the judge whether the answer follows
   from the sources the agent cited.

Running retrieval separately from the agent is deliberate. It lets us attribute a
failure to the right layer: bad retrieval versus bad generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.service import AgentService
from app.eval.dataset import EvalCase
from app.eval.faithfulness import FaithfulnessResult, judge_faithfulness
from app.eval import metrics
from app.rag.retriever import Retriever


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    got_no_answer: bool
    behaviour_bucket: str
    retrieved_doc_ids: list[str] = field(default_factory=list)
    hit: float | None = None
    recall: float | None = None
    precision: float | None = None
    reciprocal_rank: float | None = None
    answer: str = ""
    contains_expected: bool | None = None
    faithfulness: FaithfulnessResult | None = None
    flagged: bool = False


@dataclass
class EvalReport:
    cases: list[CaseResult]
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    retrieval_scored_count: int
    behaviour: metrics.BehaviourCounts
    faithfulness_mean: float
    faithfulness_judged_count: int
    flagged_count: int
    answer_match_rate: float


def _sources_text(citations) -> str:
    return "\n\n".join(
        f"[{c.marker}] ({c.document_title}) {c.snippet}" for c in citations
    )


class Evaluator:
    def __init__(
        self, retriever: Retriever, agent: AgentService, top_k: int
    ) -> None:
        self._retriever = retriever
        self._agent = agent
        self._top_k = top_k

    async def _evaluate_case(self, case: EvalCase) -> CaseResult:
        # 1. Retrieval (independent of generation).
        outcome = await self._retriever.retrieve(case.question, top_k=self._top_k)
        retrieved_ids = [c.document_id for c in outcome.retrieved]
        relevant = set(case.relevant_doc_ids)

        result = CaseResult(
            id=case.id,
            category=case.category,
            question=case.question,
            got_no_answer=False,
            behaviour_bucket="",
            retrieved_doc_ids=retrieved_ids,
        )
        if relevant:
            result.hit = metrics.hit_at_k(retrieved_ids, relevant)
            result.recall = metrics.recall_at_k(retrieved_ids, relevant)
            result.precision = metrics.precision_at_k(retrieved_ids, relevant)
            result.reciprocal_rank = metrics.reciprocal_rank(retrieved_ids, relevant)

        # 2. Behaviour: ask the agent.
        response = await self._agent.answer(case.question)
        result.got_no_answer = response.no_answer
        result.answer = response.answer
        result.behaviour_bucket = metrics.score_behaviour(
            case.expected_no_answer, response.no_answer
        )

        if case.answer_must_include and not response.no_answer:
            low = response.answer.lower()
            result.contains_expected = all(
                s.lower() in low for s in case.answer_must_include
            )

        # 3. Faithfulness for answered cases.
        if not response.no_answer and response.citations:
            result.faithfulness = await judge_faithfulness(
                case.question, response.answer, _sources_text(response.citations)
            )

        # Flag as a responsible-AI concern if the system answered an unanswerable
        # question (false answer) or produced an answer the judge could not fully
        # ground.
        result.flagged = result.behaviour_bucket == "false_answer" or (
            result.faithfulness is not None and not result.faithfulness.is_faithful
        )
        return result

    async def run(self, cases: list[EvalCase]) -> EvalReport:
        results = [await self._evaluate_case(c) for c in cases]

        retrieval_scored = [r for r in results if r.hit is not None]
        behaviour = metrics.BehaviourCounts()
        for r in results:
            setattr(
                behaviour,
                r.behaviour_bucket,
                getattr(behaviour, r.behaviour_bucket) + 1,
            )

        judged = [r for r in results if r.faithfulness is not None]
        faith_scores = [r.faithfulness.score for r in judged]

        answerable_with_expected = [
            r for r in results if r.contains_expected is not None
        ]
        matches = [r for r in answerable_with_expected if r.contains_expected]

        return EvalReport(
            cases=results,
            hit_at_k=metrics.mean([r.hit for r in retrieval_scored]),
            recall_at_k=metrics.mean([r.recall for r in retrieval_scored]),
            precision_at_k=metrics.mean([r.precision for r in retrieval_scored]),
            mrr=metrics.mean([r.reciprocal_rank for r in retrieval_scored]),
            retrieval_scored_count=len(retrieval_scored),
            behaviour=behaviour,
            faithfulness_mean=metrics.mean(faith_scores),
            faithfulness_judged_count=len(judged),
            flagged_count=sum(1 for r in results if r.flagged),
            answer_match_rate=(
                len(matches) / len(answerable_with_expected)
                if answerable_with_expected
                else 0.0
            ),
        )
