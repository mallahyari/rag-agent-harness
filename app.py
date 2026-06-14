"""Chainlit entrypoint: chainlit run app.py"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl

from config import Config
from harness.agent import run
from harness.prompts import SYSTEM_PROMPT
from harness.state import SessionState
from harness.tools import init_tools
from renderers.chainlit import ChainlitRenderer
from retrieval.vector_store import get_db

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
