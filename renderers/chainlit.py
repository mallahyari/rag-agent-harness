from __future__ import annotations

import chainlit as cl

from .base import BaseRenderer


class ChainlitRenderer(BaseRenderer):
    def __init__(self) -> None:
        self._current_step: cl.Step | None = None
        self._answer_msg: cl.Message | None = None

    async def on_thinking(self, text: str) -> None:
        async with cl.Step(name="Thinking", type="llm") as step:
            step.output = text

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        self._current_step = cl.Step(name=name, type="tool", id=tool_id)
        await self._current_step.__aenter__()

    async def on_tool_call_end(self, tool_id: str, inputs: dict) -> None:
        if self._current_step:
            self._current_step.input = inputs

    async def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        if self._current_step:
            self._current_step.output = result
            self._current_step.is_error = is_error
            await self._current_step.__aexit__(None, None, None)
            self._current_step = None

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
