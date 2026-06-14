from __future__ import annotations

import chainlit as cl

from .base import BaseRenderer


class ChainlitRenderer(BaseRenderer):
    """Chainlit renderer.

    Tool steps are sent immediately as they execute (flat, no nesting).
    The final answer is accumulated and sent as one cl.Message only after
    all steps are done — this guarantees the correct visual order:
    tool calls first, answer below.

    We avoid cl.Step context managers (.__aenter__/__aexit__) because those
    write to a contextvars.ContextVar that causes every new step to nest
    inside the previous one. Explicit .send() / .update() prevents nesting.
    """

    def __init__(self) -> None:
        self._step: cl.Step | None = None
        self._answer: str = ""

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_llm_start(self) -> None:
        pass  # No early message — we send after all steps are visible

    # ── tool events ──────────────────────────────────────────────────────────

    async def on_thinking(self, text: str) -> None:
        step = cl.Step(name="Thinking", type="llm")
        step.output = text
        await step.send()
        await step.update()

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        self._step = cl.Step(name=name, type="tool", id=tool_id)
        await self._step.send()

    async def on_tool_call_end(self, _tool_id: str, inputs: dict) -> None:
        if self._step:
            self._step.input = inputs
            await self._step.update()

    async def on_tool_result(self, _name: str, result: str, is_error: bool) -> None:
        if self._step:
            self._step.output = result
            self._step.is_error = is_error
            await self._step.update()
            self._step = None

    # ── answer events ────────────────────────────────────────────────────────

    async def on_citation_retry(self, _reason: str) -> None:
        self._answer = ""  # Discard the failed answer; accumulate the retry

    async def on_answer_start(self) -> None:
        self._answer = ""

    async def on_text_chunk(self, chunk: str) -> None:
        self._answer += chunk  # Accumulate — sent as one message in on_done

    async def on_done(self, _full_text: str) -> None:
        if self._answer:
            await cl.Message(content=self._answer).send()
        self._answer = ""
