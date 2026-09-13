"""The agent's tools: their Anthropic tool specs and their executors.

A "source" is the unifying abstraction across both tools: every fact the agent
can cite — whether a document chunk or a directory record — becomes a `Source`
with a globally-unique `[n]` marker. Markers are assigned in the order sources
are gathered across the whole run, so numbering stays stable even when the agent
calls a tool several times or mixes tools. The model cites those markers; the
finalizer resolves them back to `Source`s.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.directory import DirectoryService
from app.rag.retriever import Retriever

_SNIPPET_MAX_CHARS = 600

# --- Tool specs (Anthropic Messages API `tools` shape) -----------------------

SEARCH_TOOL_SPEC = {
    "name": "search_documents",
    "description": (
        "Search the internal company documents (the employee handbook, security "
        "policy, engineering onboarding, expense and travel policy, benefits, "
        "incident runbook, and IT FAQ) for passages relevant to a question. Use "
        "this for anything about policy, rules, benefits, process, or how-to "
        "questions. Returns numbered source passages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused natural-language search query.",
            }
        },
        "required": ["query"],
    },
}

DIRECTORY_TOOL_SPEC = {
    "name": "lookup_directory",
    "description": (
        "Look up structured organisational facts that are NOT in the policy "
        "documents: which team owns a topic, the contact channel and email, the "
        "escalation path, who is on call, and office locations. Use this for "
        "who-to-contact, which-team, what-channel, who-is-on-call, or "
        "office-address questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "A team name or topic keyword, e.g. 'security', 'IT', 'payments', 'on-call', 'offices'.",
            }
        },
        "required": ["topic"],
    },
}

TOOL_SPECS = [SEARCH_TOOL_SPEC, DIRECTORY_TOOL_SPEC]


@dataclass
class Source:
    marker: int
    source_type: str  # "document" | "directory"
    title: str
    location: str | None  # heading path (docs) or team (directory)
    reference: str  # source_path (docs) or "directory:<id>"
    snippet: str
    similarity: float | None = None


@dataclass
class ToolOutcome:
    """Result of executing one tool call."""

    result_text: str  # the text handed back to the model as a toolResult
    sources: list[Source]  # new sources gathered by this call
    step: str  # human-readable trace line


async def run_search_documents(
    retriever: Retriever, query: str, marker_start: int
) -> ToolOutcome:
    outcome = await retriever.retrieve(query)
    evidence = outcome.evidence
    sources: list[Source] = []
    lines: list[str] = []
    for offset, chunk in enumerate(evidence):
        marker = marker_start + offset
        location = chunk.heading_path or chunk.document_title
        sources.append(
            Source(
                marker=marker,
                source_type="document",
                title=chunk.document_title,
                location=chunk.heading_path,
                reference=chunk.source_path,
                snippet=chunk.content[:_SNIPPET_MAX_CHARS],
                similarity=round(chunk.similarity, 4),
            )
        )
        lines.append(f"[{marker}] (source: {chunk.document_title} — {location})\n{chunk.content}")

    if not sources:
        result_text = "No sufficiently relevant passages were found in the documents."
    else:
        result_text = "\n\n".join(lines)
    step = f"search_documents(query={query!r}) -> {len(sources)} passage(s)"
    return ToolOutcome(result_text=result_text, sources=sources, step=step)


async def run_lookup_directory(
    directory: DirectoryService, topic: str, marker_start: int
) -> ToolOutcome:
    records = directory.lookup(topic)
    sources: list[Source] = []
    lines: list[str] = []
    for offset, rec in enumerate(records):
        marker = marker_start + offset
        sources.append(
            Source(
                marker=marker,
                source_type="directory",
                title=rec.team,
                location=rec.team,
                reference=f"directory:{rec.id}",
                snippet=rec.render()[:_SNIPPET_MAX_CHARS],
                similarity=None,
            )
        )
        lines.append(f"[{marker}] (directory: {rec.team})\n{rec.render()}")

    if not sources:
        result_text = f"No directory entry was found for '{topic}'."
    else:
        result_text = "\n\n".join(lines)
    step = f"lookup_directory(topic={topic!r}) -> {len(sources)} record(s)"
    return ToolOutcome(result_text=result_text, sources=sources, step=step)
