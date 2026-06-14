from __future__ import annotations

import chainlit as cl

from .base import BaseRenderer


class ChainlitRenderer(BaseRenderer):
    """Chainlit renderer.

    Each tool call is shown as a flat, root-level Step — never nested.
    We avoid cl.Step.__aenter__/__aexit__ because those set a context variable
    that makes the next Step a child of the current one, causing deep nesting.
    Instead we use explicit .send() / .update() which don't touch the context.
    """

    def __init__(self) -> None:
        self._step: cl.Step | None = None
        self._answer_msg: cl.Message | None = None

    async def on_llm_start(self) -> None:
        pass  # Chainlit shows its own loading state per message

    async def on_thinking(self, text: str) -> None:
        step = cl.Step(name="Thinking", type="llm")
        step.output = text
        await step.send()
        await step.update()

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        # Create and send immediately so the user sees "Used X" right away.
        # Not using the context manager keeps this step at root level.
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

    async def on_citation_retry(self, _reason: str) -> None:
        pass  # The correction goes into the message history; no UI action needed

    async def on_answer_start(self) -> None:
        # Discard any in-progress message from a failed citation attempt
        if self._answer_msg:
            await self._answer_msg.remove()
        self._answer_msg = None

    async def on_text_chunk(self, chunk: str) -> None:
        if self._answer_msg is None:
            self._answer_msg = cl.Message(content="")
            await self._answer_msg.send()
        await self._answer_msg.stream_token(chunk)

    async def on_done(self, _full_text: str) -> None:
        if self._answer_msg:
            await self._answer_msg.update()
        self._answer_msg = None
