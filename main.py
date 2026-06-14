"""Terminal entrypoint for the RAG agent harness."""
from __future__ import annotations

import asyncio
import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from config import Config
from harness.agent import run
from harness.prompts import SYSTEM_PROMPT
from harness.providers import PROVIDERS
from harness.state import SessionState
from harness.tools import init_tools
from renderers.terminal import TerminalRenderer
from retrieval.vector_store import get_db


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Agent — terminal interface")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for REPL)")
    parser.add_argument(
        "--provider",
        default=None,
        choices=list(PROVIDERS.keys()),
        help="LLM provider key (default: from Config)",
    )
    args = parser.parse_args()

    config = Config(provider=args.provider) if args.provider else Config()
    db = get_db(config.lance_db_path)
    init_tools(db, config)

    renderer = TerminalRenderer()
    state = SessionState()

    if args.question:
        await run(args.question, renderer, config.provider_config, SYSTEM_PROMPT, state)
    else:
        from rich.console import Console
        console = Console()
        console.print("[bold blue]RAG Agent[/bold blue]  (type [bold]/exit[/bold] or Ctrl+C to quit)\n")
        while True:
            try:
                question = input("You: ").strip()
                if not question:
                    continue
                if question.lower() in ("/exit", "/quit"):
                    console.print("[dim]Bye.[/dim]")
                    break
                await run(
                    question, renderer, config.provider_config, SYSTEM_PROMPT, state
                )
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Bye.[/dim]")
                break


if __name__ == "__main__":
    asyncio.run(main())
