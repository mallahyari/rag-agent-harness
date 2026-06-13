from __future__ import annotations

import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
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
    "answer.text": "white",
})

_TOOL_ICONS = {
    "list_collections":  "📋",
    "search_documents":  "🔍",
    "get_context":       "📄",
    "think":             "💭",
}


class TerminalRenderer(BaseRenderer):
    def __init__(self) -> None:
        self.console = Console(theme=_THEME, highlight=False)
        self._full_text = ""
        self._streaming = False

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
        if is_error:
            title = f"[tool.err]✗ {name}[/tool.err]"
            border = "red"
        else:
            title = f"[tool.ok]✓ {name}[/tool.ok]"
            border = "green"

        preview = result[:800] + ("…" if len(result) > 800 else "")
        self.console.print(Panel(
            Text(preview),
            title=title,
            border_style=border,
            padding=(0, 1),
        ))

    async def on_answer_start(self) -> None:
        self._full_text = ""
        self._streaming = False

    async def on_text_chunk(self, chunk: str) -> None:
        self._full_text += chunk

    async def on_done(self, _full_text: str) -> None:
        if self._full_text:
            self.console.print()
            self.console.rule("[rule.answer]Answer[/rule.answer]", style="blue")
            self.console.print()
            self.console.print(Markdown(self._full_text))
            self.console.print()
        self._full_text = ""
        self._streaming = False
