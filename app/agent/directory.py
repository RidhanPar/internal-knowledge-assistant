"""Structured organisational directory — the agent's second tool.

This is deliberately *not* the document corpus. It holds structured facts —
team owners, contact channels, escalation paths, on-call, office addresses — that
you would not write as prose policy and that benefit from exact lookup rather
than semantic search. Having a second, differently-shaped source is what makes
tool *routing* a real decision for the agent, not a formality.

Backed by a small JSON file so the data is easy to edit and review. In a real
deployment this would be a table or an API (HR system, service catalogue).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class DirectoryRecord:
    id: str
    team: str
    aliases: list[str]
    owner: str
    contact_channel: str
    email: str
    escalation: str
    on_call: str
    location: str
    notes: str

    def render(self) -> str:
        """Compact, model-friendly rendering of the record's facts."""
        return (
            f"Team: {self.team}\n"
            f"Owner: {self.owner}\n"
            f"Contact channel: {self.contact_channel}\n"
            f"Email: {self.email}\n"
            f"Escalation: {self.escalation}\n"
            f"On-call: {self.on_call}\n"
            f"Location: {self.location}\n"
            f"Notes: {self.notes}"
        )


class DirectoryService:
    def __init__(self, records: list[DirectoryRecord]) -> None:
        self._records = records

    @classmethod
    def from_json(cls, path: str | Path) -> "DirectoryService":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = [DirectoryRecord(**row) for row in data]
        return cls(records)

    def lookup(self, topic: str, limit: int = 3) -> list[DirectoryRecord]:
        """Match a topic against team names and aliases by whole words.

        Scoring, highest first:
          3  the whole query equals a team name or an alias exactly
          2+ every word of a multi-word alias appears in the query
          1+ some query words match alias words (more overlap ranks higher)

        Matching is on whole words, not raw substrings. Substring matching gave
        false hits (the alias "it" is inside "security"), so an "it" question and
        a "security" question would collide. Returns an empty list when nothing
        matches, which is the signal the agent uses to fall back to no-answer.
        """
        phrase = topic.strip().lower()
        if not phrase:
            return []
        q_tokens = set(_tokens(phrase))
        if not q_tokens:
            return []

        scored: list[tuple[int, DirectoryRecord]] = []
        for rec in self._records:
            names = [rec.team, *rec.aliases]
            phrases = {n.strip().lower() for n in names}
            alias_words = {tok for n in names for tok in _tokens(n)}

            score = 0
            if phrase in phrases:
                score = 3
            else:
                for n in names:
                    n_tokens = set(_tokens(n))
                    if len(n_tokens) > 1 and n_tokens <= q_tokens:
                        score = max(score, 2 + len(n_tokens))  # full phrase present
                overlap = q_tokens & alias_words
                if overlap:
                    score = max(score, 1 + len(overlap))
            if score:
                scored.append((score, rec))

        scored.sort(key=lambda s: s[0], reverse=True)
        return [rec for _, rec in scored[:limit]]
