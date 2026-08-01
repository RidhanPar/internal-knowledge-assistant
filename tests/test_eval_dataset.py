"""Dataset integrity. Runs without Bedrock: it only reads files.

A wrong label silently corrupts every metric computed from it, so we check the
dataset points at documents and directory records that actually exist, and that
each category is internally consistent.
"""

from __future__ import annotations

from app.agent.directory import DirectoryService
from app.eval.dataset import VALID_CATEGORIES, load_dataset
from app.ingestion.loader import load_corpus


def _corpus_ids() -> set[str]:
    return {d.id for d in load_corpus("corpus")}


def _directory_ids() -> set[str]:
    svc = DirectoryService.from_json("data/directory.json")
    return {r.id for r in svc._records}  # noqa: SLF001 - test introspection


def test_dataset_loads_and_ids_are_unique():
    cases = load_dataset()
    assert len(cases) >= 12
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_categories_valid():
    for c in load_dataset():
        assert c.category in VALID_CATEGORIES


def test_relevant_doc_ids_exist_in_corpus():
    corpus_ids = _corpus_ids()
    for c in load_dataset():
        for doc_id in c.relevant_doc_ids:
            assert doc_id in corpus_ids, f"{c.id} references unknown doc {doc_id}"


def test_relevant_directory_ids_exist():
    dir_ids = _directory_ids()
    for c in load_dataset():
        for rec_id in c.relevant_directory_ids:
            assert rec_id in dir_ids, f"{c.id} references unknown directory {rec_id}"


def test_category_consistency():
    for c in load_dataset():
        if c.category == "unanswerable":
            assert c.expected_no_answer is True
            assert not c.relevant_doc_ids
        if c.category == "policy":
            assert c.expected_no_answer is False
            assert c.relevant_doc_ids, f"{c.id} is policy but labels no documents"
        if c.category == "directory":
            assert c.expected_no_answer is False
            assert c.relevant_directory_ids, f"{c.id} is directory but labels no records"


def test_has_enough_unanswerable_cases():
    # The unanswerable cases are how we measure hallucination; make sure there
    # are several, not one.
    cases = load_dataset()
    unanswerable = [c for c in cases if c.category == "unanswerable"]
    assert len(unanswerable) >= 4
