from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from harness.providers import PROVIDERS, ProviderConfig


@dataclass
class Config:
    # LLM
    provider: str = "gemini-fast"

    # Retrieval
    search_k: int = 10
    rerank_top_k: int = 5
    context_window_size: int = 2
    max_retrieved_tokens: int = 8_000

    # Harness
    max_iterations: int = 10
    max_tokens: int = 4_096

    # Paths
    lance_db_path: str = "./corpus.lance"
    bm25_index_path: str = "./corpus"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def provider_config(self) -> ProviderConfig:
        pc = PROVIDERS[self.provider]
        # Apply max_tokens from config
        from dataclasses import replace
        return replace(pc, max_tokens=self.max_tokens)
