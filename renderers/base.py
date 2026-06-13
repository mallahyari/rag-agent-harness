from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRenderer(ABC):
    @abstractmethod
    async def on_thinking(self, text: str) -> None:
        """Called when a thinking block arrives (Anthropic extended thinking)."""

    @abstractmethod
    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        """Called when the model begins a tool call."""

    @abstractmethod
    async def on_tool_call_end(self, tool_id: str, inputs: dict) -> None:
        """Called when the tool call input is fully streamed."""

    @abstractmethod
    async def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        """Called after the tool executes and returns a result."""

    @abstractmethod
    async def on_answer_start(self) -> None:
        """Called just before text chunks begin streaming for a new answer attempt."""

    @abstractmethod
    async def on_text_chunk(self, chunk: str) -> None:
        """Called for each streamed text token in the final answer."""

    @abstractmethod
    async def on_done(self, full_text: str) -> None:
        """Called when the agent loop completes."""
