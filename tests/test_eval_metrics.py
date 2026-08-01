"""Retrieval and behaviour metric math. Pure, no I/O."""

from __future__ import annotations

from app.eval import metrics


def test_hit_at_k():
    assert metrics.hit_at_k(["a", "b"], {"b"}) == 1.0
    assert metrics.hit_at_k(["a", "b"], {"z"}) == 0.0
    assert metrics.hit_at_k([], {"a"}) == 0.0
    assert metrics.hit_at_k(["a"], set()) == 0.0  # nothing labelled relevant


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], {"a", "b"}) == 1.0
    assert metrics.recall_at_k(["a", "x"], {"a", "b"}) == 0.5
    assert metrics.recall_at_k(["x"], {"a", "b"}) == 0.0


def test_precision_at_k():
    assert metrics.precision_at_k(["a", "b", "x", "y"], {"a", "b"}) == 0.5
    assert metrics.precision_at_k(["a"], {"a"}) == 1.0
    assert metrics.precision_at_k([], {"a"}) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert metrics.reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert metrics.reciprocal_rank(["x"], {"a"}) == 0.0


def test_duplicate_doc_ids_do_not_break_precision():
    # Two chunks from the same relevant doc: precision counts both retrieved slots.
    assert metrics.precision_at_k(["a", "a", "x"], {"a"}) == 2 / 3


def test_score_behaviour_buckets():
    assert metrics.score_behaviour(expected_no_answer=False, got_no_answer=False) == "answered_correctly"
    assert metrics.score_behaviour(expected_no_answer=False, got_no_answer=True) == "false_refusal"
    assert metrics.score_behaviour(expected_no_answer=True, got_no_answer=True) == "refused_correctly"
    assert metrics.score_behaviour(expected_no_answer=True, got_no_answer=False) == "false_answer"


def test_behaviour_counts_rates():
    b = metrics.BehaviourCounts(
        answered_correctly=8, false_refusal=1, refused_correctly=4, false_answer=2
    )
    assert b.total == 15
    assert b.should_answer == 9
    assert b.should_refuse == 6
    assert abs(b.false_answer_rate - 2 / 6) < 1e-9
    assert abs(b.false_refusal_rate - 1 / 9) < 1e-9
    assert abs(b.behaviour_accuracy - 12 / 15) < 1e-9


def test_mean_empty_is_zero():
    assert metrics.mean([]) == 0.0
