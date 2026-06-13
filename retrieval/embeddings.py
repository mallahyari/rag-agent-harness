from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

_embed_model: SentenceTransformer | None = None
_rerank_model: CrossEncoder | None = None


def _get_embed_model(model_name: str) -> SentenceTransformer:
    global _embed_model
    if _embed_model is None or _embed_model.model_card_data.base_model != model_name:
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _get_rerank_model(model_name: str) -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(model_name)
    return _rerank_model


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Embed a list of texts, returning a 2-D float32 array."""
    model = _get_embed_model(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors, dtype=np.float32)


def rerank(query: str, candidates: list[str], model_name: str) -> list[float]:
    """Return cross-encoder scores for each (query, candidate) pair."""
    model = _get_rerank_model(model_name)
    pairs = [(query, c) for c in candidates]
    scores: list[float] = model.predict(pairs).tolist()
    return scores
