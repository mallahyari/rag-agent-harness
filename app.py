"""Chainlit entrypoint: chainlit run app.py"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl

from config import Config
from harness.agent import run
from harness.state import SessionState
from harness.tools import init_tools
from renderers.chainlit import ChainlitRenderer
from retrieval.vector_store import get_db

SYSTEM_PROMPT = """\
You are a precise research assistant. You answer questions exclusively from the \
provided document corpus using the available tools.

TOOLS:
- list_collections: discover available document collections
- search_documents: hybrid BM25+semantic search, returns chunk_ids
- get_context(chunk_id, window=2): fetch a chunk and its neighbors for deeper reading
- think: structured reasoning before answering

RULES:
1. Always call list_collections first if you don't know what collections exist.
2. Search before answering — never answer from memory.
3. Cite every factual claim with [chunk_id] from your retrieved results.
4. If you can't find an answer after 3 searches, say so explicitly.
5. Use get_context to read more of a document when a chunk is too short to answer fully.
"""

_db = None


@cl.on_chat_start
async def on_start() -> None:
    global _db
    config = Config()
    if _db is None:
        _db = get_db(config.lance_db_path)
        init_tools(_db, config)
    cl.user_session.set("state", SessionState())
    cl.user_session.set("config", config)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    state: SessionState = cl.user_session.get("state")
    config: Config = cl.user_session.get("config")
    renderer = ChainlitRenderer()
    await run(message.content, renderer, config.provider_config, SYSTEM_PROMPT, state)
