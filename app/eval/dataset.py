"""The evaluation dataset: questions with the behaviour we expect.

Each case records what a correct system should do, so quality is measured against
a fixed reference instead of eyeballed. There are three kinds of case:

- category "policy": the answer is in the document corpus. We record which
  documents are relevant, so we can score retrieval.
- category "directory": the answer is in the structured directory, not the docs.
  Retrieval over documents is not expected to help here.
- category "unanswerable": nothing in the corpus or directory supports an answer.
  The correct behaviour is to refuse. These cases are how we measure whether the
  system hallucinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_CATEGORIES = {"policy", "directory", "unanswerable"}


@dataclass
class EvalCase:
    id: str
    question: str
    category: str
    expected_no_answer: bool
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_directory_ids: list[str] = field(default_factory=list)
    answer_must_include: list[str] = field(default_factory=list)


def load_dataset(path: str | Path = "eval/dataset.json") -> list[EvalCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [
        EvalCase(
            id=row["id"],
            question=row["question"],
            category=row["category"],
            expected_no_answer=row["expected_no_answer"],
            relevant_doc_ids=row.get("relevant_doc_ids", []),
            relevant_directory_ids=row.get("relevant_directory_ids", []),
            answer_must_include=row.get("answer_must_include", []),
        )
        for row in rows
    ]
    return cases
