from __future__ import annotations

import chainlit as cl

from .base import BaseRenderer


class ChainlitRenderer(BaseRenderer):
    """Chainlit renderer.

    One cl.Message is created per agent run (in on_llm_start) and reused for
    the entire question-answer cycle.  All tool steps are attached to it via
    parent_id so Chainlit renders them ABOVE the answer text — giving the
    correct visual order: tool calls first, answer below.

    We use explicit step.send() / step.update() (not the context-manager form)
    to prevent cl.Step from touching the context variable, which previously
    caused every step to nest inside the previous one.
    """

    def __init__(self) -> None:
        self._msg: cl.Message | None = None
        self._step: cl.Step | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_llm_start(self) -> None:
        # Create the answer container once, before any tool calls or text.
        # Steps sent afterwards use parent_id=self._msg.id so they appear
        # above the answer text in Chainlit's UI.
        if self._msg is None:
            self._msg = cl.Message(content="")
            await self._msg.send()

    # ── tool events ──────────────────────────────────────────────────────────

    async def on_thinking(self, text: str) -> None:
        parent_id = self._msg.id if self._msg else None
        step = cl.Step(name="Thinking", type="llm", parent_id=parent_id)
        step.output = text
        await step.send()
        await step.update()

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        parent_id = self._msg.id if self._msg else None
        self._step = cl.Step(name=name, type="tool", id=tool_id, parent_id=parent_id)
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
        # Clear the streamed content so the revised answer replaces it.
        # Steps already shown above are kept — they're still accurate.
        if self._msg:
            self._msg.content = ""
            await self._msg.update()

    async def on_answer_start(self) -> None:
        # Message was already created in on_llm_start; nothing to do here.
        # (Content resets for citation retries happen in on_citation_retry.)
        pass

    async def on_text_chunk(self, chunk: str) -> None:
        if self._msg is None:
            # Fallback: agent went straight to answering without a tool call
            self._msg = cl.Message(content="")
            await self._msg.send()
        await self._msg.stream_token(chunk)

    async def on_done(self, _full_text: str) -> None:
        if self._msg:
            await self._msg.update()
        self._msg = None
