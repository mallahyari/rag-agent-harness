from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class RawChunk:
    chunk_id: str
    doc_id: str
    collection: str
    text: str


def _doc_id(path: str) -> str:
    # Sanitize filename into a safe ID (strip spaces and special chars)
    stem = Path(path).stem
    return re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:80]


def _chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}__c{index:04d}"


def _read_pdf(path: str) -> str:
    from liteparse import LiteParse
    parser = LiteParse()
    result = parser.parse(path)
    return result.text


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def chunk_text(
    text: str,
    doc_id: str,
    collection: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RawChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    parts = splitter.split_text(text)
    return [
        RawChunk(
            chunk_id=_chunk_id(doc_id, i),
            doc_id=doc_id,
            collection=collection,
            text=part,
        )
        for i, part in enumerate(parts)
    ]


def chunk_file(
    path: str,
    collection: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RawChunk]:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        text = _read_pdf(path)
    else:
        text = _read_text(path)
    doc_id = _doc_id(path)
    return chunk_text(text, doc_id, collection, chunk_size, chunk_overlap)


def chunk_directory(
    directory: str,
    collection: str,
    glob: str = "**/*.pdf",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for p in sorted(Path(directory).glob(glob)):
        chunks.extend(chunk_file(str(p), collection, chunk_size, chunk_overlap))
    return chunks
