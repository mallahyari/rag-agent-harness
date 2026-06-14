from __future__ import annotations

import chainlit as cl

from .base import BaseRenderer


class ChainlitRenderer(BaseRenderer):
    """Chainlit renderer.

    Each tool call is rendered as a flat cl.Step using the context-manager
    protocol (async with), which is what makes Chainlit order it correctly on
    the timeline (steps first, answer below) and set its start/end timestamps.

    The catch: cl.Step's context manager nests each step under whatever step is
    currently open (via the local_steps contextvar). The agent fires
    on_tool_call_start for EVERY tool call in a turn before any result comes
    back, so opening steps there would nest parallel calls inside each other.

    Solution: don't open the step at call-start. Buffer name+input, then open
    AND close the whole step inside on_tool_result. The agent processes results
    sequentially (await in a for-loop), so each step fully closes before the
    next opens — local_steps never accumulates and nothing nests.

    The answer streams token-by-token into a cl.Message created on the first
    text chunk, which (in the final iteration) is after every tool step.
    """

    def __init__(self) -> None:
        self._pending: dict[str, dict] = {}  # tool_id -> {name, input}
        self._answer_msg: cl.Message | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_llm_start(self) -> None:
        pass  # Chainlit shows its own per-message loading state

    # ── tool events ──────────────────────────────────────────────────────────

    async def on_thinking(self, text: str) -> None:
        async with cl.Step(name="Thinking", type="llm") as step:
            step.output = text

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        # Buffer only — the step is created in on_tool_result to avoid nesting.
        self._pending[tool_id] = {"name": name, "input": None}

    async def on_tool_call_end(self, tool_id: str, inputs: dict) -> None:
        if tool_id in self._pending:
            self._pending[tool_id]["input"] = inputs

    async def on_tool_result(
        self, tool_id: str, name: str, result: str, is_error: bool
    ) -> None:
        info = self._pending.pop(tool_id, {"name": name, "input": None})
        async with cl.Step(name=info["name"], type="tool", id=tool_id) as step:
            if info["input"] is not None:
                step.input = info["input"]
            step.output = result
            step.is_error = is_error

    # ── answer events ────────────────────────────────────────────────────────

    async def on_citation_retry(self, _reason: str) -> None:
        # Drop the rejected answer so the revised one replaces it cleanly.
        if self._answer_msg:
            await self._answer_msg.remove()
        self._answer_msg = None

    async def on_answer_start(self) -> None:
        # If a previous attempt left a message (e.g. citation retry), discard it.
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
        self._pending.clear()
