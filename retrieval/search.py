from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index, BM25Result
from .embeddings import embed, rerank
from .vector_store import SearchResult, similarity_search


@dataclass
class HybridResult:
    chunk_id: str
    doc_id: str
    collection: str
    text: str
    score: float  # final score after RRF (and optional reranking)


def _rrf(
    semantic: list[SearchResult],
    keyword: list[BM25Result],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over two ranked lists, keyed by chunk_id."""
    scores: dict[str, float] = {}
    for rank, r in enumerate(semantic):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(keyword):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(
    db,
    collection: str,
    bm25_index: BM25Index,
    query: str,
    embed_model: str,
    k: int = 10,
) -> list[HybridResult]:
    """Hybrid BM25 + semantic search with RRF fusion."""
    query_vec = embed([query], embed_model)[0].tolist()
    sem_results = similarity_search(db, collection, query_vec, k=k)
    kw_results = bm25_index.search(query, k=k)

    rrf_scores = _rrf(sem_results, kw_results)

    # Build lookup from chunk_id → SearchResult for text/doc_id
    lookup: dict[str, SearchResult] = {r.chunk_id: r for r in sem_results}

    # Chunks only in BM25 results won't be in lookup — fetch them from the store if needed
    # For simplicity we only return chunks we have full data for
    results: list[HybridResult] = []
    for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        if chunk_id in lookup:
            r = lookup[chunk_id]
            results.append(HybridResult(
                chunk_id=chunk_id,
                doc_id=r.doc_id,
                collection=r.collection,
                text=r.text,
                score=score,
            ))
    return results[:k]


def rerank_results(
    query: str,
    results: list[HybridResult],
    rerank_model: str,
    top_k: int,
) -> list[HybridResult]:
    """Re-score with a cross-encoder and return top_k."""
    if not results:
        return []
    texts = [r.text for r in results]
    scores = rerank(query, texts, rerank_model)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    out = []
    for r, s in ranked[:top_k]:
        out.append(HybridResult(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            collection=r.collection,
            text=r.text,
            score=float(s),
        ))
    return out


def format_results(results: list[HybridResult]) -> str:
    """Format search results as a structured string for the LLM."""
    if not results:
        return "(no results found)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{r.chunk_id}] (doc: {r.doc_id}, score: {r.score:.3f})\n{r.text}"
        )
    return "\n\n---\n\n".join(lines)
