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
        ids = re.findall(r'\[([^\]]+)\]', result)
        state.retrieved_chunk_ids.update(ids)

        if "(no results found)" in result:
            return result + (
                "\n\n[Hint: No results. Try a shorter or more general query, "
                "or a different search term.]"
            )

    if tool_name == "get_context" and not is_error:
        state.retrieved_token_count += len(result) // 4
        # Track chunk IDs returned by get_context so citation validator accepts them
        ids = re.findall(r'\[([^\]]+)\]', result)
        state.retrieved_chunk_ids.update(ids)

    return result


_NO_INFO_PHRASES = (
    "cannot find", "can't find", "not found", "no information",
    "not present", "not available", "unable to find", "not in the",
    "not contain", "does not contain", "no results",
)


def validate_citation(answer: str, retrieved_ids: set[str]) -> str | None:
    cited = set(re.findall(r'\[([^\]]+)\]', answer))

    # If the model explicitly says it couldn't find info, no citations required
    if not cited:
        lower = answer.lower()
        if any(p in lower for p in _NO_INFO_PHRASES) or not retrieved_ids:
            return None
        return (
            "Your answer has no citations. Add [chunk_id] references "
            "for each factual claim using the IDs from search results."
        )

    # Only reject hallucinated IDs — ones not in any retrieved result
    unknown = cited - retrieved_ids
    if unknown:
        return (
            f"You cited {unknown} which were not in your retrieved results. "
            "Only cite chunks you actually retrieved."
        )
    return None
