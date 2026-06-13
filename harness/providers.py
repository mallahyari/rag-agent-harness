from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Any

import litellm

litellm.set_verbose = False


@dataclass
class ProviderConfig:
    model: str
    max_tokens: int = 4096
    enable_thinking: bool = False
    thinking_budget: int = 5000


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic-fast":  ProviderConfig(model="claude-haiku-4-5-20251001"),
    "anthropic-smart": ProviderConfig(model="claude-sonnet-4-6", enable_thinking=True),
    "openai-fast":     ProviderConfig(model="gpt-4o-mini"),
    "openai-smart":    ProviderConfig(model="gpt-4o"),
    "gemini-fast":     ProviderConfig(model="gemini/gemini-2.5-flash"),
    "gemini-smart":    ProviderConfig(model="gemini/gemini-2.5-pro"),
}


async def stream_completion(
    messages: list[dict],
    tools: list[dict],
    config: ProviderConfig,
    system: str,
) -> Any:
    extra: dict[str, Any] = {}

    if config.enable_thinking and "claude" in config.model:
        extra["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }

    full_messages = [{"role": "system", "content": system}] + messages

    response = await litellm.acompletion(
        model=config.model,
        messages=full_messages,
        tools=tools if tools else None,
        stream=True,
        max_tokens=config.max_tokens,
        **extra,
    )
    return response
