"""Claude generation via the Anthropic API (direct).

We call the Anthropic Messages API rather than Bedrock. It is the same Claude
model family, reached directly with an API key, which avoids the AWS account
authorization that blocked Bedrock.

The Claude 5-era API does not expose a temperature setting (it was removed in
favour of effort levels). For grounded, cited answers we rely on a strict system
prompt and short, extraction-style outputs rather than a temperature knob.

The API is used two ways:
- `generate`: a single system + user turn, for the single-shot answer path and
  the evaluation judge.
- `converse_with_tools`: a multi-turn loop where the model may return tool_use
  blocks instead of a final answer. The agent runs the tools and feeds
  tool_result blocks back in. We keep the raw assistant message so it can be
  appended verbatim to the running conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import anthropic

from app.config import get_settings
from app.core.errors import UpstreamError


@lru_cache
def _client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    # An organization-scoped key needs the workspace id in a header. A
    # workspace-scoped key does not, so we only send it when configured.
    headers = (
        {"anthropic-workspace-id": settings.anthropic_workspace_id}
        if settings.anthropic_workspace_id
        else None
    )
    # Passing None lets the SDK fall back to the ANTHROPIC_API_KEY env var.
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key or None,
        default_headers=headers,
    )


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


def _text_from(content) -> str:
    return "".join(block.text for block in content if block.type == "text").strip()


async def generate(system_prompt: str, user_prompt: str) -> LLMResult:
    settings = get_settings()
    try:
        resp = await _client().messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIError as exc:
        raise UpstreamError("anthropic-generation", str(exc)) from exc
    return LLMResult(
        text=_text_from(resp.content),
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason or "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool-using conversation (the agentic layer)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class ConverseResult:
    text: str
    tool_uses: list[ToolUse]
    stop_reason: str
    raw_message: dict  # the assistant message, to append to the conversation
    input_tokens: int
    output_tokens: int


async def converse_with_tools(
    system_prompt: str, messages: list[dict], tool_specs: list[dict]
) -> ConverseResult:
    settings = get_settings()
    try:
        resp = await _client().messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            system=system_prompt,
            messages=messages,
            tools=tool_specs,
        )
    except anthropic.APIError as exc:
        raise UpstreamError("anthropic-generation", str(exc)) from exc

    tool_uses = [
        ToolUse(id=b.id, name=b.name, input=b.input)
        for b in resp.content
        if b.type == "tool_use"
    ]
    # Rebuild the assistant message as plain dicts so it can be appended to the
    # message list and sent back on the next turn.
    raw_message = {
        "role": "assistant",
        "content": [b.model_dump() for b in resp.content],
    }
    return ConverseResult(
        text=_text_from(resp.content),
        tool_uses=tool_uses,
        stop_reason=resp.stop_reason or "",
        raw_message=raw_message,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
