"""Faithfulness judging: does the answer actually follow from its sources?

Retrieval metrics tell us whether the right context was fetched. Faithfulness
tells us whether the generated answer stayed inside that context or drifted into
claims the sources do not support. An answer can cite real sources and still be
unfaithful if it adds a detail the sources never state, so this is a separate
check from retrieval.

We judge with a second, independent LLM call (LLM as judge). The judge sees only
the question, the answer, and the source passages, and is told to use no outside
knowledge. It returns a verdict and the specific claims it could not find support
for. This is the standard way to scale groundedness checking beyond hand review,
and it is itself an example of critically evaluating model output rather than
trusting it.

Limitations we are honest about: the judge is a model and can be wrong, so we
run it at temperature 0, keep its prompt strict, and surface the unsupported
claims it names so a human can check. A stronger setup would use a different
model family as judge and calibrate it against human labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.rag.llm import generate

_VERDICT_SCORES = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge. You are given a \
QUESTION, an ANSWER, and the SOURCE passages the answer is supposed to be based \
on. Decide whether every factual statement in the ANSWER is supported by the \
SOURCES. Use no outside knowledge. A statement counts as supported only if the \
SOURCES state it or clearly imply it.

Respond with a single JSON object and nothing else, using exactly these keys:
  "verdict": one of "supported", "partial", "unsupported"
  "unsupported_claims": array of strings, each a claim from the ANSWER not \
supported by the SOURCES (empty array if none)
  "reasoning": one short sentence

Use "supported" when every claim is backed by the sources, "partial" when some \
claims are backed and at least one is not, and "unsupported" when the core claim \
is not backed."""


@dataclass
class FaithfulnessResult:
    verdict: str  # "supported" | "partial" | "unsupported" | "unknown"
    score: float
    unsupported_claims: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw: str = ""

    @property
    def is_faithful(self) -> bool:
        return self.verdict == "supported"


def _extract_json_object(text: str) -> dict | None:
    """Pull the first top-level JSON object out of the model text.

    Models sometimes wrap JSON in prose or code fences even when told not to, so
    we do not assume the whole string parses.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_judge_response(text: str) -> FaithfulnessResult:
    """Parse the judge's reply into a result. Never raises: an unparseable reply
    becomes an 'unknown' verdict so one bad judge call does not crash a run."""
    obj = _extract_json_object(text)
    if obj is None:
        return FaithfulnessResult(verdict="unknown", score=0.0, raw=text)

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in _VERDICT_SCORES:
        return FaithfulnessResult(
            verdict="unknown",
            score=0.0,
            unsupported_claims=list(obj.get("unsupported_claims", []) or []),
            reasoning=str(obj.get("reasoning", "")),
            raw=text,
        )

    return FaithfulnessResult(
        verdict=verdict,
        score=_VERDICT_SCORES[verdict],
        unsupported_claims=list(obj.get("unsupported_claims", []) or []),
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


def build_judge_prompt(question: str, answer: str, sources_text: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        "Return only the JSON object."
    )


async def judge_faithfulness(
    question: str, answer: str, sources_text: str
) -> FaithfulnessResult:
    """Run the judge model over one answer and return its verdict."""
    prompt = build_judge_prompt(question, answer, sources_text)
    result = await generate(JUDGE_SYSTEM_PROMPT, prompt)
    return parse_judge_response(result.text)
