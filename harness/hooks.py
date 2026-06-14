from __future__ import annotations

import re

from .state import SessionState

MAX_RETRIEVED_TOKENS = 8_000


class ContextBudgetExceeded(Exception):
    pass


def pre_hook(tool_name: str, args: dict, state: SessionState) -> dict:
    if tool_name in ("search_documents", "get_context"):
        if state.retrieved_token_count > MAX_RETRIEVED_TOKENS:
            raise ContextBudgetExceeded(
                "Context budget reached. Synthesize from what you have already retrieved."
            )
    return args


def post_hook(tool_name: str, result: str, is_error: bool, state: SessionState) -> str:
    if tool_name == "search_documents" and not is_error:
        # Match only the chunk-ID header lines: "[chunk_id] (doc: …)" — not bracket
        # patterns inside document text (e.g. [1], [Smith 2023], [Table 3]).
        ids = re.findall(r'^\[([^\]]+)\] \(doc:', result, re.MULTILINE)
        state.retrieved_chunk_ids.update(ids)

        if "(no results found)" in result:
            return result + (
                "\n\n[Hint: No results. Try a shorter or more general query, "
                "or a different search term.]"
            )

    if tool_name == "get_context" and not is_error:
        state.retrieved_token_count += len(result) // 4
        # get_context format: "[chunk_id]\ntext" — match IDs on their own line
        ids = re.findall(r'^\[([^\]]+)\]$', result, re.MULTILINE)
        state.retrieved_chunk_ids.update(ids)

    return result


_NO_INFO_PHRASES = (
    "cannot find", "can't find", "not found", "no information",
    "not present", "not available", "unable to find", "not in the",
    "not contain", "does not contain", "no results",
)


def _parse_cited_ids(answer: str) -> set[str]:
    """Extract citation IDs from an answer, normalising [id1, id2] multi-cites."""
    raw = re.findall(r'\[([^\]]+)\]', answer)
    result: set[str] = set()
    for item in raw:
        # Split comma/semicolon-separated multi-cites into individual IDs
        for part in re.split(r'[,;]\s*', item):
            part = part.strip()
            if part:
                result.add(part)
    return result


def validate_citation(answer: str, retrieved_ids: set[str]) -> str | None:
    cited = _parse_cited_ids(answer)

    # If the model explicitly says it couldn't find info, no citations required
    if not cited:
        lower = answer.lower()
        if any(p in lower for p in _NO_INFO_PHRASES) or not retrieved_ids:
            return None
        valid_sample = ", ".join(sorted(retrieved_ids)[:6])
        return (
            "Your answer has no citations. For every factual claim add the chunk_id "
            "in square brackets immediately after it, e.g. [chunk_id]. "
            f"Valid chunk IDs from your search results: {valid_sample}"
        )

    # Only reject hallucinated IDs — ones not in any retrieved result
    unknown = cited - retrieved_ids
    if unknown:
        bad = ", ".join(sorted(unknown))
        good = ", ".join(sorted(retrieved_ids)[:6])
        return (
            f"Citation error: {bad} — those IDs were not in your retrieved results. "
            f"Valid chunk IDs you can cite: {good}. "
            "Put each chunk ID in its own square brackets: [chunk_id1] [chunk_id2]."
        )
    return None
