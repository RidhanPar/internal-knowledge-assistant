"""Retrieval and behaviour metrics. Pure functions, no I/O, so they are cheap to
unit test and the definitions are unambiguous.

Retrieval metrics compare an ordered list of retrieved document ids against the
set of documents we labelled relevant for a question.

- hit_at_k: did at least one relevant document appear in the top k? This is the
  most forgiving retrieval metric. If it is low, the generator never even had the
  right context, so a wrong answer is retrieval's fault, not the model's.

- recall_at_k: what fraction of the relevant documents appeared in the top k?
  This matters when a question needs several documents. With one relevant
  document, recall is either 0.0 or 1.0 and equals hit.

- precision_at_k: what fraction of the retrieved chunks were relevant? Low
  precision means we are feeding the model noise, which both raises cost and
  gives it room to ground on the wrong passage.

- reciprocal_rank: 1 divided by the rank of the first relevant document (1.0 if
  it is first, 0.5 if second, 0 if absent). Averaged across questions this is
  Mean Reciprocal Rank (MRR). It rewards putting the right document near the top,
  which is what the generator reads first.

Behaviour metrics score whether the system answered when it should and refused
when it should, which is the responsible-AI part: a system that answers an
unanswerable question is hallucinating.
"""

from __future__ import annotations

from dataclasses import dataclass


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(r in relevant_ids for r in retrieved_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    found = {r for r in retrieved_ids if r in relevant_ids}
    return len(found) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    found = [r for r in retrieved_ids if r in relevant_ids]
    return len(found) / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class BehaviourCounts:
    """Confusion-matrix style counts for the answer/refuse decision.

    "should answer" means the corpus supports an answer; "should refuse" means it
    does not. A false answer (refused=False on a should-refuse case) is the
    hallucination case we most want to avoid.
    """

    answered_correctly: int = 0  # should answer, did answer
    false_refusal: int = 0       # should answer, refused
    refused_correctly: int = 0   # should refuse, refused
    false_answer: int = 0        # should refuse, answered

    @property
    def total(self) -> int:
        return (
            self.answered_correctly
            + self.false_refusal
            + self.refused_correctly
            + self.false_answer
        )

    @property
    def should_answer(self) -> int:
        return self.answered_correctly + self.false_refusal

    @property
    def should_refuse(self) -> int:
        return self.refused_correctly + self.false_answer

    @property
    def behaviour_accuracy(self) -> float:
        if not self.total:
            return 0.0
        return (self.answered_correctly + self.refused_correctly) / self.total

    @property
    def false_answer_rate(self) -> float:
        """Of the questions that should be refused, how many were answered.
        This is the hallucination rate for unanswerable questions."""
        return self.false_answer / self.should_refuse if self.should_refuse else 0.0

    @property
    def false_refusal_rate(self) -> float:
        """Of the questions that should be answered, how many were refused."""
        return self.false_refusal / self.should_answer if self.should_answer else 0.0


def score_behaviour(expected_no_answer: bool, got_no_answer: bool) -> str:
    """Classify one case into a behaviour bucket."""
    if expected_no_answer:
        return "refused_correctly" if got_no_answer else "false_answer"
    return "false_refusal" if got_no_answer else "answered_correctly"
