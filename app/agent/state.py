"""Graph state for the knowledge agent.

We carry the full Bedrock Converse `messages` list through the graph so the
tool-use round-trip has perfect fidelity, plus the accumulated `sources` (for
citation resolution) and a `steps` trace (for observability). Nodes return
complete updated lists rather than relying on reducers — explicit and easy to
follow.
"""

from __future__ import annotations

from typing import TypedDict

from app.agent.tools import Source
from app.rag.llm import ToolUse


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict]
    sources: list[Source]
    steps: list[str]
    iterations: int
    max_iterations: int
    # Set by the agent node when the model requested tools; consumed by the
    # tools node and then cleared.
    pending_tool_uses: list[ToolUse]
    # Set by the agent node when the model produced a final answer instead.
    final_text: str | None
    stop_reason: str
    # Token accounting across all model calls in the run.
    input_tokens: int
    output_tokens: int
    # Set by the finalize node.
    no_answer: bool
    used_markers: list[int]
