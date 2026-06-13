from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

from rank_bm25 import BM25Okapi


class BM25Result(NamedTuple):
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        tokenized = [t.lower().split() for t in texts]
        self._chunk_ids = chunk_ids
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> list[BM25Result]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            BM25Result(chunk_id=self._chunk_ids[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self._chunk_ids, "bm25": self._bm25}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["ids"]
        self._bm25 = data["bm25"]

    @property
    def size(self) -> int:
        return len(self._chunk_ids)


_indexes: dict[str, BM25Index] = {}


def get_index(collection: str) -> BM25Index:
    if collection not in _indexes:
        _indexes[collection] = BM25Index()
    return _indexes[collection]


def add_to_index(collection: str, chunk_ids: list[str], texts: list[str]) -> None:
    idx = get_index(collection)
    existing_ids = list(idx._chunk_ids)
    # Rebuild with all documents (BM25Okapi requires full corpus at build time)
    all_ids = existing_ids + chunk_ids
    # We can't recover old texts after build, so the index is rebuilt in pipeline.py
    # This function is only called during initial ingestion
    idx.build(all_ids if existing_ids else chunk_ids,
              [""] * len(existing_ids) + texts if existing_ids else texts)


def rebuild_index(collection: str, chunk_ids: list[str], texts: list[str]) -> BM25Index:
    idx = BM25Index()
    idx.build(chunk_ids, texts)
    _indexes[collection] = idx
    return idx


def save_index(collection: str, base_path: str) -> None:
    path = str(Path(base_path).with_suffix("")) + f"_{collection}.bm25"
    get_index(collection).save(path)


def load_index(collection: str, base_path: str) -> BM25Index:
    path = str(Path(base_path).with_suffix("")) + f"_{collection}.bm25"
    idx = BM25Index()
    idx.load(path)
    _indexes[collection] = idx
    return idx


def load_or_empty(collection: str, base_path: str) -> BM25Index:
    path = str(Path(base_path).with_suffix("")) + f"_{collection}.bm25"
    if Path(path).exists():
        return load_index(collection, base_path)
    return get_index(collection)
