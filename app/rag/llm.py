"""Claude generation via the Bedrock Converse API.

We use `converse` rather than raw `invoke_model` because it is model-agnostic
(the same call shape works across Claude versions and even other providers),
which keeps BEDROCK_LLM_MODEL_ID a pure configuration switch.

Generation runs at temperature 0: for grounded, cited answers we want the most
faithful, least creative response, and reproducibility during evaluation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.config import get_settings
from app.rag.bedrock import get_bedrock_runtime


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


def _converse_sync(system_prompt: str, user_prompt: str) -> LLMResult:
    settings = get_settings()
    client = get_bedrock_runtime()
    resp = client.converse(
        modelId=settings.bedrock_llm_model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={
            "maxTokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        },
    )
    message = resp["output"]["message"]
    text = "".join(block.get("text", "") for block in message["content"]).strip()
    usage = resp.get("usage", {})
    return LLMResult(
        text=text,
        input_tokens=usage.get("inputTokens", 0),
        output_tokens=usage.get("outputTokens", 0),
        stop_reason=resp.get("stopReason", ""),
    )


async def generate(system_prompt: str, user_prompt: str) -> LLMResult:
    return await asyncio.to_thread(_converse_sync, system_prompt, user_prompt)
