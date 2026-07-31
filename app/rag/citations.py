"""Citation-marker parsing, shared by the single-shot and agent answer paths.

Citations are derived from the `[n]` markers the model *actually wrote*, not from
whatever was retrieved — so the citation list reflects grounding, not just recall.
"""

from __future__ import annotations

import re

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def cited_markers(answer: str, n_sources: int) -> list[int]:
    """Distinct, in-range `[n]` markers, in order of first appearance."""
    seen: list[int] = []
    for m in _MARKER_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen
