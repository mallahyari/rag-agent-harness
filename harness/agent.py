from __future__ import annotations

import json
import re

from .hooks import ContextBudgetExceeded, post_hook, pre_hook, validate_citation
from .providers import ProviderConfig, stream_completion
from .state import SessionState
from .tools import TOOL_SCHEMAS, dispatch_tool
from renderers.base import BaseRenderer

MAX_ITERATIONS = 10


def _clean_tool_id(tid: str) -> str:
    # Gemini thinking mode appends '__thought__<base64>' to tool call IDs.
    # Strip it so history round-trips cleanly.
    return re.split(r"__thought__", tid, maxsplit=1)[0]


async def run(
    question: str,
    renderer: BaseRenderer,
    config: ProviderConfig,
    system: str,
    state: SessionState,
) -> str:
    state.history.append({"role": "user", "content": question})

    for iteration in range(MAX_ITERATIONS):
        stream = await stream_completion(
            messages=state.history,
            tools=TOOL_SCHEMAS,
            config=config,
            system=system,
        )

        full_text = ""
        tool_calls: dict[str, dict] = {}  # tool_id → {name, input}
        last_tid: str | None = None

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # Anthropic extended thinking
            if hasattr(delta, "thinking") and delta.thinking:
                await renderer.on_thinking(delta.thinking)

            # Streaming text
            if delta.content:
                full_text += delta.content
                await renderer.on_text_chunk(delta.content)

            # Tool call streaming
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id:
                        last_tid = _clean_tool_id(tc.id)
                        tool_calls[last_tid] = {
                            "name": tc.function.name or "",
                            "input": "",
                        }
                        await renderer.on_tool_call_start(
                            tc.function.name or "", last_tid
                        )
                    elif last_tid:
                        pass  # continuation of current tool call

                    # Accumulate argument JSON fragment
                    raw_tid = tc.id or (last_tid or "")
                    tid = _clean_tool_id(raw_tid) if raw_tid else last_tid
                    if tid and tc.function and tc.function.arguments:
                        tool_calls[tid]["input"] += tc.function.arguments
                        if tc.function.name and not tool_calls[tid]["name"]:
                            tool_calls[tid]["name"] = tc.function.name

        finish_reason = chunk.choices[0].finish_reason

        if finish_reason == "stop" or (finish_reason is None and not tool_calls):
            correction = validate_citation(full_text, state.retrieved_chunk_ids)
            if correction:
                state.history.append({"role": "assistant", "content": full_text})
                state.history.append({"role": "user", "content": correction})
                continue
            await renderer.on_done(full_text)
            return full_text

        if finish_reason == "tool_calls" or tool_calls:
            # Parse all inputs first
            parsed: dict[str, dict] = {}
            for tid, tc in tool_calls.items():
                try:
                    parsed[tid] = json.loads(tc["input"] or "{}")
                except json.JSONDecodeError:
                    parsed[tid] = {}
                await renderer.on_tool_call_end(tid, parsed[tid])

            # Store in OpenAI tool_calls format — works across all LiteLLM providers
            assistant_msg: dict = {
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": [
                    {
                        "id": tid,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(parsed[tid]),
                        },
                    }
                    for tid, tc in tool_calls.items()
                ],
            }
            state.history.append(assistant_msg)

            tool_results: list[dict] = []
            for tid, tc in tool_calls.items():
                inputs = parsed[tid]
                try:
                    args = pre_hook(tc["name"], inputs, state)
                    result, is_error = dispatch_tool(tc["name"], args)
                except ContextBudgetExceeded as e:
                    result, is_error = str(e), True

                result = post_hook(tc["name"], result, is_error, state)
                await renderer.on_tool_result(tc["name"], result, is_error)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": result,
                })

            state.history.extend(tool_results)

    await renderer.on_done("Iteration limit reached without a final answer.")
    return "Iteration limit reached."
