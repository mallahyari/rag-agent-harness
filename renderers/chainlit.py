from __future__ import annotations

import chainlit as cl
from chainlit.step import utc_now

from .base import BaseRenderer


class ChainlitRenderer(BaseRenderer):
    """Chainlit renderer.

    Tool steps are sent immediately as they execute (flat, no nesting) and are
    tracked in a dict keyed by tool_id so that PARALLEL tool calls in one
    iteration each receive their own result — otherwise only the last step
    would get an output and the rest would render empty.

    IMPORTANT: we set step.start / step.end manually. The Chainlit frontend
    orders the timeline by these timestamps; the context-manager form
    (__aenter__/__aexit__) sets them automatically, but we can't use it because
    it nests every step under the previous one. Using send()/update() without
    setting start/end leaves them None, which makes steps sort incorrectly
    relative to the final answer message — the bug that put the answer on top.

    The final answer is accumulated and sent as ONE cl.Message in on_done, after
    every step exists and with the latest timestamp, so it renders last.
    """

    def __init__(self) -> None:
        self._steps: dict[str, cl.Step] = {}  # tool_id -> Step
        self._answer: str = ""

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def on_llm_start(self) -> None:
        pass  # The answer message is created last (on_done), never here

    # ── tool events ──────────────────────────────────────────────────────────

    async def on_thinking(self, text: str) -> None:
        step = cl.Step(name="Thinking", type="llm")
        step.start = utc_now()
        step.output = text
        step.end = utc_now()
        await step.send()
        await step.update()

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        step = cl.Step(name=name, type="tool", id=tool_id)
        step.start = utc_now()
        await step.send()
        self._steps[tool_id] = step

    async def on_tool_call_end(self, tool_id: str, inputs: dict) -> None:
        step = self._steps.get(tool_id)
        if step:
            step.input = inputs
            await step.update()

    async def on_tool_result(
        self, tool_id: str, _name: str, result: str, is_error: bool
    ) -> None:
        step = self._steps.get(tool_id)
        if step:
            step.output = result
            step.is_error = is_error
            step.end = utc_now()
            await step.update()

    # ── answer events ────────────────────────────────────────────────────────

    async def on_citation_retry(self, _reason: str) -> None:
        self._answer = ""  # Discard the rejected answer; accumulate the retry

    async def on_answer_start(self) -> None:
        self._answer = ""

    async def on_text_chunk(self, chunk: str) -> None:
        self._answer += chunk  # Accumulate; sent once in on_done

    async def on_done(self, _full_text: str) -> None:
        if self._answer:
            await cl.Message(content=self._answer).send()
        self._answer = ""
        self._steps.clear()
