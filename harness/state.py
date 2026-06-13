from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    history: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: set[str] = field(default_factory=set)
    retrieved_token_count: int = 0
