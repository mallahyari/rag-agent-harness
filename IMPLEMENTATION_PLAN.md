# RAG Agent Harness — Implementation Plan

---

## Tech Stack Decisions

### LLM Provider: LiteLLM
Multi-provider support via a single unified interface. LiteLLM normalizes streaming events, tool call formats, and error types across Anthropic, OpenAI, and Gemini. Switching providers is a one-line config change.

```python
# Same code, different provider
litellm.acompletion(model="claude-haiku-4-5-20251001", ...)   # Anthropic
litellm.acompletion(model="gpt-4o-mini", ...)                 # OpenAI
litellm.acompletion(model="gemini/gemini-2.0-flash", ...)     # Gemini
```

Tool call format follows OpenAI standard — LiteLLM converts it for each provider transparently.

### UI: Renderer abstraction
The harness emits events. A `BaseRenderer` defines the interface. `TerminalRenderer` (Rich) and `ChainlitRenderer` implement it. The harness has no knowledge of which renderer is active. Switch via `--ui` flag or entrypoint.

### Vector Store: LanceDB
Embedded, disk-based, no server. Supports hybrid search. Single file on disk.

### Search: Hybrid (BM25 + semantic) + Cross-encoder reranking
`rank-bm25` for keyword search, `sentence-transformers` for embeddings and reranking.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   User Question      │
                        └────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      ENTRYPOINT          │
                    │  main.py (terminal)      │
                    │  app.py (chainlit)       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     RENDERER             │
                    │  TerminalRenderer        │◄── Rich live panels
                    │  ChainlitRenderer        │◄── Chainlit Steps + stream
                    └────────────┬────────────┘
                                 │  events
                    ┌────────────▼────────────┐
                    │     AGENT HARNESS        │
                    │  agent.py                │
                    │  ┌──────────────────┐    │
                    │  │  LiteLLM call    │    │◄── providers.py
                    │  │  stream events   │    │
                    │  │  tool dispatch   │    │
                    │  │  hooks           │    │◄── hooks.py
                    │  │  history[]       │    │
                    │  └──────────────────┘    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     TOOL REGISTRY        │
                    │  tools.py                │
                    │  list_collections        │
                    │  search_documents        │
                    │  get_context             │
                    │  think                   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   RETRIEVAL PIPELINE     │
                    │  search.py               │
                    │  hybrid search (RRF)     │
                    │  cross-encoder reranking │
                    │  vector_store.py         │◄── LanceDB
                    │  bm25.py                 │◄── rank-bm25
                    │  embeddings.py           │◄── sentence-transformers
                    └─────────────────────────┘
```

---

## File Structure

```
rag-harness/
│
├── harness/
│   ├── agent.py          # Core agent loop — streaming ReAct
│   ├── providers.py      # LiteLLM wrapper, provider config, model registry
│   ├── tools.py          # Tool implementations + JSON schemas
│   ├── hooks.py          # Pre/post middleware (budget, hints, citation check)
│   └── state.py          # Session state: history, context budget, retrieved chunk IDs
│
├── retrieval/
│   ├── search.py         # hybrid_search(), rerank(), format_results()
│   ├── vector_store.py   # LanceDB wrapper: insert, search, get_neighbors
│   ├── bm25.py           # BM25 index: build, persist, search
│   └── embeddings.py     # Embedding model wrapper (local or API)
│
├── ingestion/
│   ├── chunker.py        # Document → chunks (recursive, with overlap)
│   ├── pipeline.py       # Full ingestion: load → chunk → embed → index
│   └── ingest.py         # CLI: python -m ingestion.ingest --path ./docs
│
├── renderers/
│   ├── base.py           # BaseRenderer — abstract interface
│   ├── terminal.py       # Rich: live panels, spinners, tool call boxes
│   └── chainlit.py       # Chainlit: Steps, stream_token, thinking panels
│
├── config.py             # All settings: provider, model, k, timeouts, paths
├── main.py               # Terminal entrypoint
├── app.py                # Chainlit entrypoint (chainlit run app.py)
└── pyproject.toml
```

---

## Component Contracts

### `BaseRenderer` — `renderers/base.py`

All renderer methods are async. The harness calls these; it never touches Rich or Chainlit directly.

```python
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
    async def on_text_chunk(self, chunk: str) -> None:
        """Called for each streamed text token in the final answer."""

    @abstractmethod
    async def on_done(self, full_text: str) -> None:
        """Called when the agent loop completes."""
```

### `TerminalRenderer` — `renderers/terminal.py`

Uses `rich.live.Live` with a `Layout` that shows:
- Top panel: current tool call (name + inputs streaming in)
- Middle panel: tool result (collapsed after 3s)
- Bottom panel: streaming final answer

```python
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from .base import BaseRenderer

class TerminalRenderer(BaseRenderer):
    def __init__(self):
        self.console = Console()

    async def on_thinking(self, text: str) -> None:
        self.console.print(Panel(text, title="[dim]Thinking[/dim]", border_style="dim"))

    async def on_tool_call_start(self, name: str, tool_id: str) -> None:
        self.console.print(f"\n[bold cyan]→ {name}[/bold cyan]", end="")

    async def on_tool_call_end(self, tool_id: str, inputs: dict) -> None:
        import json
        self.console.print(f"  [dim]{json.dumps(inputs)[:120]}[/dim]")

    async def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        style = "red" if is_error else "green"
        self.console.print(Panel(
            result[:500],
            title=f"[{style}]← {name}[/{style}]",
            border_style=style,
        ))

    async def on_text_chunk(self, chunk: str) -> None:
        self.console.print(chunk, end="", markup=False)

    async def on_done(self, full_text: str) -> None:
        self.console.print("\n")
```

### `ChainlitRenderer` — `renderers/chainlit.py`

```python
import chainlit as cl
from .base import BaseRenderer

class ChainlitRenderer(BaseRenderer):
    def __init__(self):
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

    async def on_text_chunk(self, chunk: str) -> None:
        if self._answer_msg is None:
            self._answer_msg = cl.Message(content="")
            await self._answer_msg.send()
        await self._answer_msg.stream_token(chunk)

    async def on_done(self, full_text: str) -> None:
        if self._answer_msg:
            await self._answer_msg.update()
```

---

### `providers.py` — LiteLLM wrapper

```python
import litellm
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    model: str                    # e.g. "claude-haiku-4-5-20251001", "gpt-4o-mini"
    max_tokens: int = 4096
    enable_thinking: bool = False # Anthropic extended thinking only
    thinking_budget: int = 5000   # tokens allocated to thinking

PROVIDERS = {
    "anthropic-fast":  ProviderConfig(model="claude-haiku-4-5-20251001"),
    "anthropic-smart": ProviderConfig(model="claude-sonnet-4-6", enable_thinking=True),
    "openai-fast":     ProviderConfig(model="gpt-4o-mini"),
    "openai-smart":    ProviderConfig(model="gpt-4o"),
    "gemini-fast":     ProviderConfig(model="gemini/gemini-2.0-flash"),
    "gemini-smart":    ProviderConfig(model="gemini/gemini-2.5-pro"),
}

async def stream_completion(
    messages: list[dict],
    tools: list[dict],
    config: ProviderConfig,
    system: str,
):
    """Unified streaming call across providers."""
    extra = {}

    # Anthropic extended thinking (provider-specific)
    if config.enable_thinking and "claude" in config.model:
        extra["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }

    # LiteLLM uses OpenAI message format — add system as first message
    full_messages = [{"role": "system", "content": system}] + messages

    response = await litellm.acompletion(
        model=config.model,
        messages=full_messages,
        tools=tools,
        stream=True,
        max_tokens=config.max_tokens,
        **extra,
    )
    return response
```

---

### `agent.py` — Core Agent Loop

The loop is provider-agnostic and renderer-agnostic. It only knows about `providers.py` and `BaseRenderer`.

```python
from .providers import ProviderConfig, stream_completion
from .state import SessionState
from .hooks import pre_hook, post_hook, validate_citation
from .tools import TOOL_SCHEMAS, dispatch_tool
from renderers.base import BaseRenderer

MAX_ITERATIONS = 10

async def run(
    question: str,
    renderer: BaseRenderer,
    config: ProviderConfig,
    system: str,
    state: SessionState,
) -> str:
    state.history.append({"role": "user", "content": question})

    for iteration in range(MAX_ITERATIONS):
        stream = await stream_completion(
            messages=state.history,
            tools=TOOL_SCHEMAS,
            config=config,
            system=system,
        )

        full_text = ""
        tool_calls = {}   # tool_id → {name, input_chunks}

        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Thinking block (Anthropic only, surfaced via LiteLLM extra)
            if hasattr(delta, "thinking") and delta.thinking:
                await renderer.on_thinking(delta.thinking)

            # Streaming text
            if delta.content:
                full_text += delta.content
                await renderer.on_text_chunk(delta.content)

            # Tool call streaming
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tid = tc.id or list(tool_calls)[-1]  # ongoing call uses last id
                    if tc.id:  # new tool call starting
                        tool_calls[tid] = {"name": tc.function.name, "input": ""}
                        await renderer.on_tool_call_start(tc.function.name, tid)
                    if tc.function.arguments:
                        tool_calls[tid]["input"] += tc.function.arguments

        # Assess stop reason
        finish_reason = chunk.choices[0].finish_reason

        if finish_reason == "stop":
            # Validate citations before returning
            correction = validate_citation(full_text, state.retrieved_chunk_ids)
            if correction:
                state.history.append({"role": "assistant", "content": full_text})
                state.history.append({"role": "user", "content": correction})
                continue  # let the model revise
            await renderer.on_done(full_text)
            return full_text

        if finish_reason == "tool_calls":
            # Reconstruct assistant message with tool_use blocks
            import json
            assistant_content = []
            if full_text:
                assistant_content.append({"type": "text", "text": full_text})
            for tid, tc in tool_calls.items():
                inputs = json.loads(tc["input"] or "{}")
                await renderer.on_tool_call_end(tid, inputs)
                assistant_content.append({
                    "type": "tool_use", "id": tid,
                    "name": tc["name"], "input": inputs,
                })
            state.history.append({"role": "assistant", "content": assistant_content})

            # Execute tools
            tool_results = []
            for tid, tc in tool_calls.items():
                inputs = json.loads(tc["input"] or "{}")
                args = pre_hook(tc["name"], inputs, state)
                result, is_error = dispatch_tool(tc["name"], args)
                result = post_hook(tc["name"], result, is_error, state)
                await renderer.on_tool_result(tc["name"], result, is_error)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result,
                })
            state.history.extend(tool_results)

    return "Iteration limit reached."
```

---

### `state.py` — Session State

```python
from dataclasses import dataclass, field

@dataclass
class SessionState:
    history: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: set[str] = field(default_factory=set)
    retrieved_token_count: int = 0
```

### `hooks.py` — Middleware

```python
from .state import SessionState

MAX_RETRIEVED_TOKENS = 8_000

def pre_hook(tool_name: str, args: dict, state: SessionState) -> dict:
    if tool_name in ("search_documents", "get_context"):
        if state.retrieved_token_count > MAX_RETRIEVED_TOKENS:
            raise ContextBudgetExceeded(
                "Context budget reached. Synthesize from what you have already retrieved."
            )
    return args

def post_hook(tool_name: str, result: str, is_error: bool, state: SessionState) -> str:
    if tool_name == "search_documents":
        if not is_error:
            # Track chunk IDs for citation validation
            import re
            ids = re.findall(r'\[([^\]]+)\]', result)
            state.retrieved_chunk_ids.update(ids)

        if "(no results found)" in result:
            return result + (
                "\n\n[Hint: No results. Try a shorter or more general query, "
                "or a different search term.]"
            )

    if tool_name == "get_context" and not is_error:
        # Rough token estimate (4 chars ≈ 1 token)
        state.retrieved_token_count += len(result) // 4

    return result

def validate_citation(answer: str, retrieved_ids: set[str]) -> str | None:
    import re
    cited = set(re.findall(r'\[([^\]]+)\]', answer))
    if not cited:
        return (
            "Your answer has no citations. Add [chunk_id] references "
            "for each factual claim using the IDs from search results."
        )
    unknown = cited - retrieved_ids
    if unknown:
        return (
            f"You cited {unknown} which were not in your retrieved results. "
            "Only cite chunks you actually retrieved."
        )
    return None

class ContextBudgetExceeded(Exception):
    pass
```

---

### `config.py`

```python
from dataclasses import dataclass
from harness.providers import PROVIDERS, ProviderConfig

@dataclass
class Config:
    # LLM
    provider: str = "anthropic-fast"     # key into PROVIDERS dict
    
    # Retrieval
    search_k: int = 10                   # candidates before reranking
    rerank_top_k: int = 5                # after reranking
    context_window_size: int = 2         # chunks before/after in get_context
    max_retrieved_tokens: int = 8_000

    # Harness
    max_iterations: int = 10
    max_tokens: int = 4_096

    # Paths
    lance_db_path: str = "./corpus.lance"
    bm25_index_path: str = "./corpus.bm25"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local, no API key needed
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def provider_config(self) -> ProviderConfig:
        return PROVIDERS[self.provider]
```

---

### Entrypoints

**`main.py` — Terminal**
```python
import asyncio
import argparse
from harness.agent import run
from harness.state import SessionState
from harness.providers import PROVIDERS
from renderers.terminal import TerminalRenderer
from config import Config

SYSTEM_PROMPT = "..."  # from RAG_AGENT_HARNESS.md

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--provider", default="anthropic-fast", choices=PROVIDERS.keys())
    args = parser.parse_args()

    config = Config(provider=args.provider)
    renderer = TerminalRenderer()
    state = SessionState()

    if args.question:
        await run(args.question, renderer, config.provider_config, SYSTEM_PROMPT, state)
    else:
        # REPL mode
        print("RAG Agent (Ctrl+C to exit)\n")
        while True:
            try:
                question = input("You: ").strip()
                if question:
                    await run(question, renderer, config.provider_config, SYSTEM_PROMPT, state)
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    asyncio.run(main())
```

**`app.py` — Chainlit**
```python
import chainlit as cl
from harness.agent import run
from harness.state import SessionState
from harness.providers import PROVIDERS
from renderers.chainlit import ChainlitRenderer
from config import Config

SYSTEM_PROMPT = "..."

@cl.on_chat_start
async def on_start():
    config = Config()
    cl.user_session.set("state", SessionState())
    cl.user_session.set("config", config)

@cl.on_message
async def on_message(message: cl.Message):
    state: SessionState = cl.user_session.get("state")
    config: Config = cl.user_session.get("config")
    renderer = ChainlitRenderer()
    await run(message.content, renderer, config.provider_config, SYSTEM_PROMPT, state)
```

Run with:
```bash
python main.py "What does the report say about Q3 revenue?"          # terminal
python main.py --provider openai-fast "..."                          # different provider
chainlit run app.py                                                  # browser UI
```

---

## Build Order

### Phase 1 — Retrieval core (no LLM yet)
```
retrieval/embeddings.py     embed a sentence, verify output shape
retrieval/vector_store.py   insert 10 chunks, run similarity search
retrieval/bm25.py           build index, run keyword search
retrieval/search.py         hybrid_search() + rerank() end to end
ingestion/chunker.py        chunk 3 documents, inspect output
ingestion/pipeline.py       full ingest of a small corpus
```
**Test:** `python -m ingestion.ingest --path ./sample_docs` → verify chunks in LanceDB.

### Phase 2 — Terminal agent, single provider
```
harness/state.py
harness/providers.py        LiteLLM, Anthropic only first
harness/tools.py            all 4 tools wired to retrieval
harness/hooks.py            pre/post hooks, citation validator
harness/agent.py            streaming loop, TerminalRenderer events
renderers/base.py
renderers/terminal.py       Rich panels, minimal version
main.py                     single-question mode
```
**Test:** `python main.py "your question"` — see tool calls + streaming answer in terminal.

### Phase 3 — Multi-provider + Chainlit
```
harness/providers.py        add OpenAI + Gemini configs
renderers/chainlit.py       Chainlit Steps + stream
app.py
```
**Test:** Switch `--provider openai-fast`, compare answers. `chainlit run app.py` in browser.

### Phase 4 — Polish
```
REPL mode in main.py        persistent session history across questions
Config CLI flags             --provider, --k, --thinking
Extended thinking toggle     enable for anthropic-smart provider
Evaluation script            run 20 HotpotQA questions, measure citation rate
```

---

## Dependencies

```toml
[project]
name = "rag-harness"
requires-python = ">=3.11"

dependencies = [
    # LLM
    "litellm>=1.40",
    "anthropic>=0.40",           # for extended thinking extras

    # Retrieval
    "lancedb>=0.12",
    "sentence-transformers>=3.0",
    "rank-bm25>=0.2",

    # Ingestion
    "langchain-text-splitters>=0.3",

    # UI
    "chainlit>=2.0",
    "rich>=13.0",

    # Utilities
    "python-dotenv>=1.0",
]
```

```
# .env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

---

## What Each Phase Delivers

| Phase | What you can do |
|---|---|
| 1 | Ingest documents, run raw search queries, inspect chunk quality |
| 2 | Ask questions in terminal, see tool calls + streaming answer with Rich |
| 3 | Same in browser (Chainlit), switch providers with one flag |
| 4 | Multi-turn sessions, evaluate retrieval quality on HotpotQA |
