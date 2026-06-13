from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Tool schemas (OpenAI format, LiteLLM-compatible)
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_collections",
            "description": (
                "List all available document collections in the corpus. "
                "Call this first to know what collections exist before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document corpus using hybrid BM25 + semantic search. "
                "Returns ranked chunks with their chunk_ids. "
                "Use these chunk_ids when citing sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection to search. Use list_collections to discover available collections.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 20).",
                        "default": 5,
                    },
                },
                "required": ["query", "collection"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": (
                "Retrieve a specific chunk and its surrounding context window. "
                "Use this to read more of a document after finding a relevant chunk_id via search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id from a search result.",
                    },
                    "collection": {
                        "type": "string",
                        "description": "The collection this chunk belongs to.",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Number of chunks before/after to include (default 2).",
                        "default": 2,
                    },
                },
                "required": ["chunk_id", "collection"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Use this tool to reason step-by-step before answering. "
                "Especially useful for multi-hop questions that require combining "
                "information from multiple chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning, notes, or plan.",
                    },
                },
                "required": ["thought"],
            },
        },
    },
]


# Runtime dependencies injected at startup
_db: Any = None
_bm25_indexes: dict[str, Any] = {}
_config: Any = None


def init_tools(db: Any, config: Any) -> None:
    global _db, _config
    _db = db
    _config = config


def dispatch_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool by name. Returns (result_text, is_error)."""
    try:
        if name == "list_collections":
            return _list_collections()
        elif name == "search_documents":
            return _search_documents(**args)
        elif name == "get_context":
            return _get_context(**args)
        elif name == "think":
            return _think(**args)
        else:
            return f"Unknown tool: {name}", True
    except Exception as e:
        return f"Tool error ({name}): {e}", True


def _list_collections() -> tuple[str, bool]:
    from retrieval.vector_store import list_collections
    cols = list_collections(_db)
    if not cols:
        return "No collections found. Ingest documents first.", False
    return "Available collections:\n" + "\n".join(f"  - {c}" for c in cols), False


def _search_documents(query: str, collection: str, k: int = 5) -> tuple[str, bool]:
    from retrieval.bm25 import load_or_empty
    from retrieval.search import format_results, hybrid_search, rerank_results

    k = min(k, 20)
    bm25_idx = load_or_empty(collection, _config.bm25_index_path)

    candidates = hybrid_search(
        db=_db,
        collection=collection,
        bm25_index=bm25_idx,
        query=query,
        embed_model=_config.embedding_model,
        k=_config.search_k,
    )
    reranked = rerank_results(
        query=query,
        results=candidates,
        rerank_model=_config.reranker_model,
        top_k=min(k, _config.rerank_top_k),
    )
    return format_results(reranked), False


def _get_context(chunk_id: str, collection: str, window: int = 2) -> tuple[str, bool]:
    from retrieval.vector_store import get_chunk, get_neighbors

    target = get_chunk(_db, collection, chunk_id)
    if target is None:
        return f"Chunk '{chunk_id}' not found in collection '{collection}'.", True

    neighbors = get_neighbors(_db, collection, chunk_id, window)

    # Sort all by chunk_id (encodes position)
    all_chunks = sorted([target] + neighbors, key=lambda c: c.chunk_id)
    parts = [f"[{c.chunk_id}]\n{c.text}" for c in all_chunks]
    return "\n\n---\n\n".join(parts), False


def _think(thought: str) -> tuple[str, bool]:
    return f"Thought recorded:\n{thought}", False
