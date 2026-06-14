from __future__ import annotations

import re

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.status import Status
from rich.theme import Theme

from .base import BaseRenderer

_THEME = Theme({
    "tool.name":     "bold cyan",
    "tool.args":     "dim white",
    "tool.ok":       "green",
    "tool.err":      "bold red",
    "tool.result":   "dim white",
    "thinking":      "italic yellow",
    "rule.answer":   "bold blue",
    "llm.thinking":  "dim cyan",
})

_TOOL_ICONS = {
    "list_collections": "📋",
    "search_documents": "🔍",
    "get_context":      "📄",
    "think":            "💭",
}

_MD_PATTERN = re.compile(r"(\*\*|#{1,3} |^\s*[-*] |\`)", re.MULTILINE)


def _has_markdown(text: str) -> bool:
    return bool(_MD_PATTERN.search(text))


def _format_args(inputs: dict) -> str:
    """Compact human-readable arg string, not a JSON blob."""
    if not inputs:
        return ""
    parts = []
    for k, v in inputs.items():
        if isinstance(v, str):
            display = v if len(v) <= 60 else v[:60] + "…"
            parts.append(f"{k}: \"{display}\"")
        else:
            parts.append(f"{k}: {v}")
    result = "  ·  ".join(parts)
    return result if len(result) <= 140 else result[:140] + "…"


def _result_summary(name: str, result: str) -> str:
    """One-line summary of a tool result for inline display."""
    if name == "list_collections":
        # Format is "Available collections:\n  - name\n  - name"
        lines = [ln[4:] for ln in result.splitlines() if ln.startswith("  - ")]
        if lines:
            return f"{len(lines)} collection(s): {', '.join(lines)}"
        return result.strip()[:120]

    if name == "search_documents":
        chunk_ids = re.findall(r'\[([^\]]+)\]', result)
        n = len(chunk_ids)
        if n == 0:
            return "no results found"
        return f"{n} chunk(s) retrieved"

    if name == "get_context":
        chunk_ids = re.findall(r'\[([^\]]+)\]', result)
        if chunk_ids:
            return "chunk " + " + ".join(chunk_ids[:5])
        return f"{len(result)} chars"

    if name == "think":
        lines = result.strip().splitlines()
        first = next((l for l in lines if l.strip()), "")
        return first[:100] or "reasoning recorded"

    return f"{len(result)} chars"


class TerminalRenderer(BaseRenderer):
    def __init__(self) -> None:
        self.console = Console(theme=_THEME, highlight=False)
        self._full_text = ""
        self._answer_header_shown = False
        self._current_tool = ""
        self._status: Status | None = None

    # ── spinner helpers ──────────────────────────────────────────────────────

    def _start_status(self, text: str, style: str = "cyan") -> None:
        self._stop_status()
        self._status = self.console.status(text, spinner="dots", spinner_style=style)
        self._status.start()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    # ── renderer events ──────────────────────────────────────────────────────

    async def on_llm_start(self) -> None:
        self._start_status("[llm.thinking]Thinking…[/llm.thinking]", style="cyan")

    async def on_thinking(self, text: str) -> None:
        self._stop_status()
        # Compact one-liner preview — Anthropic extended thinking
        first_line = text.strip().splitlines()[0][:120] if text.strip() else ""
        self.console.print(f"[thinking]◆ thinking[/thinking]  [dim]{first_line}[/dim]")

    async def on_tool_call_start(self, name: str, _tool_id: str) -> None:
        self._stop_status()
        self._current_tool = name
        icon = _TOOL_ICONS.get(name, "⚙")
        self._start_status(f"[tool.name]{icon} {name}[/tool.name]", style="cyan")

    async def on_tool_call_end(self, _tool_id: str, inputs: dict) -> None:
        self._stop_status()
        icon = _TOOL_ICONS.get(self._current_tool, "⚙")
        args_str = _format_args(inputs)
        line = f"[tool.name]{icon} {self._current_tool}[/tool.name]"
        if args_str:
            line += f"  [tool.args]{args_str}[/tool.args]"
        self.console.print(line)

    async def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        if is_error:
            preview = result[:200] + ("…" if len(result) > 200 else "")
            self.console.print(f"  [tool.err]✗ {preview}[/tool.err]")
        else:
            summary = _result_summary(name, result)
            self.console.print(f"  [tool.ok]└─[/tool.ok] [tool.result]{summary}[/tool.result]")

    async def on_citation_retry(self, reason: str) -> None:
        self._stop_status()
        # Show what the validator rejected so the user can see why a retry happened
        snippet = reason.split(".")[0][:120]
        self.console.print(f"\n[dim]  ↻ revising — {snippet}[/dim]")

    async def on_answer_start(self) -> None:
        self._stop_status()
        if not self._answer_header_shown:
            self.console.print()
            self.console.print(Rule("[rule.answer]Answer[/rule.answer]", style="blue"))
            self.console.print()
            self._answer_header_shown = True
        else:
            self.console.print()
        self._full_text = ""

    async def on_text_chunk(self, chunk: str) -> None:
        self._full_text += chunk
        self.console.print(chunk, end="", markup=False, highlight=False)

    async def on_done(self, _full_text: str) -> None:
        self._stop_status()
        if self._full_text:
            self.console.print("\n")
            if _has_markdown(self._full_text):
                self.console.print(Rule("[dim]↑ plain  ·  formatted ↓[/dim]", style="dim"))
                self.console.print()
                self.console.print(Markdown(self._full_text))
                self.console.print()
        self._full_text = ""
        self._answer_header_shown = False
