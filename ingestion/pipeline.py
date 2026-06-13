from __future__ import annotations

from pathlib import Path

from retrieval.bm25 import rebuild_index, save_index
from retrieval.embeddings import embed
from retrieval.vector_store import Chunk, get_db, insert_chunks

from .chunker import RawChunk, chunk_directory, chunk_file


def ingest(
    path: str,
    collection: str,
    lance_db_path: str,
    bm25_base_path: str,
    embed_model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    glob: str = "**/*.txt",
    batch_size: int = 64,
) -> int:
    """Ingest a file or directory into the corpus. Returns number of chunks added."""
    p = Path(path)
    if p.is_dir():
        raw_chunks = chunk_directory(str(p), collection, glob, chunk_size, chunk_overlap)
    else:
        raw_chunks = chunk_file(str(p), collection, chunk_size, chunk_overlap)

    if not raw_chunks:
        return 0

    db = get_db(lance_db_path)

    # Embed in batches
    all_chunks: list[Chunk] = []
    for i in range(0, len(raw_chunks), batch_size):
        batch: list[RawChunk] = raw_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = embed(texts, embed_model)
        for rc, vec in zip(batch, vectors):
            all_chunks.append(Chunk(
                chunk_id=rc.chunk_id,
                doc_id=rc.doc_id,
                collection=rc.collection,
                text=rc.text,
                vector=vec.tolist(),
            ))

    insert_chunks(db, all_chunks)

    # Rebuild BM25 over the entire collection (needed for consistent IDF)
    from retrieval.vector_store import _table
    tbl = _table(db, collection)
    rows = tbl.search().limit(100_000).to_list()
    all_ids = [r["chunk_id"] for r in rows]
    all_texts = [r["text"] for r in rows]
    rebuild_index(collection, all_ids, all_texts)
    save_index(collection, bm25_base_path)

    return len(all_chunks)
