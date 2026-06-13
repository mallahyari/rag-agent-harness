from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .base import BaseRenderer


class TerminalRenderer(BaseRenderer):
    def __init__(self) -> None:
        self.console = Console()
        self._answer_started = False

    async def on_thinking(self, text: str) -> None:
        self.console.print(Panel(
            Text(text, style="dim"),
            title="[dim]Thinking[/dim]",
            border_style="dim",
        ))

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        self.console.print(f"\n[bold cyan]→ {name}[/bold cyan]", end="")

    async def on_tool_call_end(self, tool_id: str, inputs: dict) -> None:
        self.console.print(f"  [dim]{json.dumps(inputs)[:160]}[/dim]")

    async def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        style = "red" if is_error else "green"
        preview = result[:600] + ("…" if len(result) > 600 else "")
        self.console.print(Panel(
            preview,
            title=f"[{style}]← {name}[/{style}]",
            border_style=style,
        ))

    async def on_text_chunk(self, chunk: str) -> None:
        if not self._answer_started:
            self.console.print()
            self.console.print(Rule("[bold]Answer[/bold]", style="blue"))
            self._answer_started = True
        self.console.print(chunk, end="", markup=False)

    async def on_done(self, full_text: str) -> None:
        if self._answer_started:
            self.console.print("\n")
        self._answer_started = False
