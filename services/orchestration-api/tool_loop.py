"""Shared bounded tool-calling loop for the two chat surfaces
(routers/chat.py and routers/portal_assistant.py).

Both need the same shape — call the model with tools, execute the
non-destructive calls it emits, feed the results back, repeat until it
answers — but differ in policy: chat.py scopes tools per persona and
stops on a destructive tool for human confirmation; portal_assistant.py
is read-only and describes a destructive tool instead of running it.
Parameterising the policy here keeps one loop instead of two copies.

Bounded by `max_rounds` so a model that keeps calling tools can't spin
forever. A destructive call stops the loop and is returned as `pending`
(never auto-executed) — the caller decides the wording and, for chat.py,
resubmits it as a confirmed call. A call to one of `local_tools` (a
synthetic tool not backed by any MCP server, e.g. ask_user_tool.py) stops
it the same way and comes back as `local_call` instead — portal_assistant.py
is the one caller today.

Streams the model call per round (ILLMGatewayAdapter.chat_completion_stream)
instead of blocking for a full response — `on_delta`, if given, is called
with each reasoning/content fragment as it arrives, which is what lets
portal_assistant.py show live "thinking" progress instead of a static
indicator for the 15-45s a reasoning model can take. chat.py doesn't pass
`on_delta` and reads the same `LoopResult.usage`/`cost_usd` shape as
before, so it needs no changes for this.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from mcp_client import McpToolRegistry, ToolSchema

from adapters.ai_platform.interfaces import ILLMGatewayAdapter

DEFAULT_MAX_ROUNDS = 8

# Called after each executed tool with (name, args, result) — lets a
# streaming caller (portal_assistant) emit an event per call.
ToolCallHook = Callable[[str, dict[str, Any], str], Awaitable[None]]

# Called per streamed fragment as the model produces it, before a round
# finishes — "reasoning" is a reasoning model's thinking tokens (not the
# visible reply), "content" is the visible reply itself (which can arrive
# on a non-final round too, e.g. preamble text before a tool call).
DeltaKind = Literal["reasoning", "content"]
OnDeltaHook = Callable[[DeltaKind, str], Awaitable[None]]


@dataclass
class LoopResult:
    # Final assistant message (the one with no further tool_calls), or the
    # last message when the round budget ran out.
    message: dict[str, Any]
    tool_calls: list[str] = field(default_factory=list)
    # Set when a destructive tool was proposed but not executed.
    pending: dict[str, Any] | None = None
    # Set when a `local_tools` entry (e.g. ask_user_tool.ASK_USER_TOOL) was
    # called — its raw arguments, never dispatched through the MCP
    # registry. Mutually exclusive with `pending`: the loop returns on the
    # first of either kind of call it hits in a round.
    local_call: dict[str, Any] | None = None
    # True only when the loop ran out of `max_rounds` while the model was
    # still mid-flow (its last round emitted more tool_calls, never a
    # final answer) — the caller's cue to say so explicitly rather than
    # show nothing or a truncated `message` with no clear explanation.
    exhausted: bool = False
    usage: dict[str, Any] | None = None
    # Always None today — LiteLLMGatewayAdapter.chat_completion_stream has
    # no cost source (see its own docstring: the response-cost header this
    # deployment was expected to send doesn't exist, streaming or not, a
    # pre-existing gap this loop doesn't attempt to fix).
    cost_usd: float | None = None


async def run_tool_loop(
    llm: ILLMGatewayAdapter,
    registry: McpToolRegistry,
    messages: list[dict[str, object]],
    *,
    model: str,
    allowed_tools: frozenset[str] | None = None,
    local_tools: dict[str, ToolSchema] | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    on_tool_call: ToolCallHook | None = None,
    on_delta: OnDeltaHook | None = None,
) -> LoopResult:
    """Run the model with `tools` until it answers or `max_rounds` is hit.

    `messages` is mutated in place (assistant + tool turns appended) so a
    caller can keep the transcript. `allowed_tools=None` exposes every
    tool; a concrete set scopes to a persona. `local_tools` (name ->
    schema) are offered alongside the MCP ones but never dispatched
    through `registry` — a call to one of them stops the loop and comes
    back as `LoopResult.local_call` instead of being executed (see
    ask_user_tool.py for the one caller uses today).
    """
    tools = registry.list_tools(allowed_tools) + list((local_tools or {}).values())
    tool_calls_made: list[str] = []
    message: dict[str, Any] = {}
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None

    for _ in range(max_rounds):
        content_parts: list[str] = []
        # Accumulated by delta index, standard OpenAI-style tool-call
        # streaming — `function.arguments` is a fragment on every delta,
        # not the full string, so it's concatenated as it arrives.
        tool_call_parts: dict[int, dict[str, Any]] = {}

        async for chunk in llm.chat_completion_stream(model=model, messages=messages, tools=tools):
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning_content")
                if reasoning and on_delta is not None:
                    await on_delta("reasoning", reasoning)
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    if on_delta is not None:
                        await on_delta("content", text)
                for tc in delta.get("tool_calls") or []:
                    slot = tool_call_parts.setdefault(
                        tc["index"],
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if "id" in tc:
                        slot["id"] = tc["id"]
                    if "type" in tc:
                        slot["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    if "name" in fn:
                        slot["function"]["name"] += fn["name"]
                    if "arguments" in fn:
                        slot["function"]["arguments"] += fn["arguments"]
            chunk_usage = chunk.get("usage")
            if chunk_usage is not None:
                usage = cast(dict[str, Any], chunk_usage)

        # Reconstructed to the exact shape ChatCompletionMessage already
        # had — every line below this point (destructive check, tool
        # execution, messages.append) is unchanged from the pre-streaming
        # version and doesn't know the message was assembled from deltas.
        message = {"role": "assistant", "content": "".join(content_parts) or None}
        ordered_calls = [tool_call_parts[i] for i in sorted(tool_call_parts)]
        if ordered_calls:
            message["tool_calls"] = ordered_calls

        calls = message.get("tool_calls") or []
        if not calls:
            return LoopResult(
                message=message, tool_calls=tool_calls_made, usage=usage, cost_usd=cost_usd
            )

        messages.append(cast(dict[str, object], message))
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            # Only a confirmed resubmission may pass "confirm" — never the model.
            args.pop("confirm", None)

            if local_tools and name in local_tools:
                return LoopResult(
                    message=message,
                    tool_calls=tool_calls_made,
                    local_call={"name": name, "arguments": args},
                    usage=usage,
                    cost_usd=cost_usd,
                )

            if registry.is_destructive(name):
                return LoopResult(
                    message=message,
                    tool_calls=tool_calls_made,
                    pending={"name": name, "arguments": args},
                    usage=usage,
                    cost_usd=cost_usd,
                )

            result = await registry.call_tool(name, args)
            tool_calls_made.append(name)
            if on_tool_call is not None:
                await on_tool_call(name, args, result)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    # Only reachable by exhausting every round in range(max_rounds) — every
    # earlier iteration returned above (no tool_calls, a local_tools call,
    # or a destructive one), so getting here means the model was still
    # calling tools on the very last round with no final answer in hand.
    return LoopResult(
        message=message, tool_calls=tool_calls_made, exhausted=True, usage=usage, cost_usd=cost_usd
    )


def last_text(message: dict[str, Any]) -> str:
    """The assistant text from a loop result, or "" when it ended on a
    tool call with no content."""
    content = message.get("content")
    return content if isinstance(content, str) else ""
