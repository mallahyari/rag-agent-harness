from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    collection: str
    text: str
    vector: list[float]


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    collection: str
    text: str
    score: float


def _table(db: Any, collection: str) -> Any:
    return db.open_table(collection)


def get_db(path: str):
    import lancedb
    return lancedb.connect(path)


def ensure_collection(db: Any, collection: str, dim: int) -> None:
    import pyarrow as pa
    if collection not in db.table_names():
        schema = pa.schema([
            pa.field("chunk_id", pa.utf8()),
            pa.field("doc_id", pa.utf8()),
            pa.field("collection", pa.utf8()),
            pa.field("text", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])
        db.create_table(collection, schema=schema)


def insert_chunks(db: Any, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    by_collection: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_collection.setdefault(c.collection, []).append(c)

    for collection, group in by_collection.items():
        dim = len(group[0].vector)
        ensure_collection(db, collection, dim)
        tbl = _table(db, collection)
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "collection": c.collection,
                "text": c.text,
                "vector": c.vector,
            }
            for c in group
        ]
        tbl.add(rows)


def similarity_search(
    db: Any,
    collection: str,
    query_vector: list[float],
    k: int,
) -> list[SearchResult]:
    if collection not in db.table_names():
        return []
    tbl = _table(db, collection)
    rows = (
        tbl.search(query_vector)
        .metric("cosine")
        .limit(k)
        .to_list()
    )
    results = []
    for row in rows:
        # LanceDB cosine returns distance (0=identical); convert to similarity
        score = 1.0 - float(row.get("_distance", 0.0))
        results.append(SearchResult(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            collection=row["collection"],
            text=row["text"],
            score=score,
        ))
    return results


def get_chunk(db: Any, collection: str, chunk_id: str) -> SearchResult | None:
    if collection not in db.table_names():
        return None
    tbl = _table(db, collection)
    rows = tbl.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_list()
    if not rows:
        return None
    row = rows[0]
    return SearchResult(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        collection=row["collection"],
        text=row["text"],
        score=1.0,
    )


def get_neighbors(
    db: Any,
    collection: str,
    chunk_id: str,
    window: int,
) -> list[SearchResult]:
    """Return up to `window` chunks before and after `chunk_id` in the same doc."""
    if collection not in db.table_names():
        return []
    tbl = _table(db, collection)
    target = get_chunk(db, collection, chunk_id)
    if target is None:
        return []

    doc_id = target.doc_id
    # Fetch all chunks for this doc, ordered by chunk_id (which encodes position)
    rows = (
        tbl.search()
        .where(f"doc_id = '{doc_id}'")
        .limit(10_000)
        .to_list()
    )
    rows.sort(key=lambda r: r["chunk_id"])
    ids = [r["chunk_id"] for r in rows]
    try:
        idx = ids.index(chunk_id)
    except ValueError:
        return []

    lo = max(0, idx - window)
    hi = min(len(rows), idx + window + 1)
    neighbors = []
    for r in rows[lo:hi]:
        if r["chunk_id"] == chunk_id:
            continue
        neighbors.append(SearchResult(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            collection=r["collection"],
            text=r["text"],
            score=1.0,
        ))
    return neighbors


def list_collections(db: Any) -> list[str]:
    return db.table_names()
