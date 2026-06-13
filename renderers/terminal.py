from __future__ import annotations

import json
import re

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from .base import BaseRenderer

_THEME = Theme({
    "tool.name":   "bold cyan",
    "tool.args":   "dim white",
    "tool.ok":     "bold green",
    "tool.err":    "bold red",
    "thinking":    "italic dim yellow",
    "rule.answer": "bold blue",
    "stream.text": "white",
})

_TOOL_ICONS = {
    "list_collections":  "📋",
    "search_documents":  "🔍",
    "get_context":       "📄",
    "think":             "💭",
}

_MD_PATTERN = re.compile(r"(\*\*|#{1,3} |^\s*[-*] |\`)", re.MULTILINE)


def _has_markdown(text: str) -> bool:
    return bool(_MD_PATTERN.search(text))


class TerminalRenderer(BaseRenderer):
    def __init__(self) -> None:
        self.console = Console(theme=_THEME, highlight=False)
        self._full_text = ""
        self._answer_header_shown = False

    async def on_thinking(self, text: str) -> None:
        self.console.print(Panel(
            Text(text, style="thinking"),
            title="[thinking]◆ Thinking[/thinking]",
            border_style="yellow",
            padding=(0, 1),
        ))

    async def on_tool_call_start(self, name: str, _tool_id: str) -> None:
        icon = _TOOL_ICONS.get(name, "⚙")
        self.console.print(f"\n[tool.name]{icon} {name}[/tool.name]", end="")

    async def on_tool_call_end(self, _tool_id: str, inputs: dict) -> None:
        if inputs:
            args_str = json.dumps(inputs, ensure_ascii=False)
            if len(args_str) > 200:
                args_str = args_str[:200] + "…"
            self.console.print(f"  [tool.args]{args_str}[/tool.args]")
        else:
            self.console.print()

    async def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        style  = "tool.err" if is_error else "tool.ok"
        icon   = "✗" if is_error else "✓"
        border = "red" if is_error else "green"
        preview = result[:800] + ("…" if len(result) > 800 else "")
        self.console.print(Panel(
            Text(preview),
            title=f"[{style}]{icon} {name}[/{style}]",
            border_style=border,
            padding=(0, 1),
        ))

    async def on_answer_start(self) -> None:
        if not self._answer_header_shown:
            self.console.print()
            self.console.print(Rule("[rule.answer]Answer[/rule.answer]", style="blue"))
            self.console.print()
            self._answer_header_shown = True
        else:
            # Retry after citation failure — show a subtle separator
            self.console.print("\n[dim]  ↻ revising…[/dim]\n")
        self._full_text = ""

    async def on_text_chunk(self, chunk: str) -> None:
        self._full_text += chunk
        # Stream each token immediately — plain text, no redraws, no flicker
        self.console.print(chunk, end="", markup=False, highlight=False)

    async def on_done(self, _full_text: str) -> None:
        if self._full_text:
            self.console.print("\n")
            # If the answer contains Markdown syntax, reprint it formatted
            if _has_markdown(self._full_text):
                self.console.print(Rule("[dim]↑ plain  ·  formatted ↓[/dim]", style="dim"))
                self.console.print()
                self.console.print(Markdown(self._full_text))
                self.console.print()
        self._full_text = ""
        self._answer_header_shown = False
