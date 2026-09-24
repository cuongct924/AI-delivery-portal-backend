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
resubmits it as a confirmed call.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from mcp_client import McpToolRegistry

from adapters.ai_platform.interfaces import ILLMGatewayAdapter

DEFAULT_MAX_ROUNDS = 5

# Called after each executed tool with (name, args, result) — lets a
# streaming caller (portal_assistant) emit an event per call.
ToolCallHook = Callable[[str, dict[str, Any], str], Awaitable[None]]


@dataclass
class LoopResult:
    # Final assistant message (the one with no further tool_calls), or the
    # last message when the round budget ran out.
    message: dict[str, Any]
    tool_calls: list[str] = field(default_factory=list)
    # Set when a destructive tool was proposed but not executed.
    pending: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None


async def run_tool_loop(
    llm: ILLMGatewayAdapter,
    registry: McpToolRegistry,
    messages: list[dict[str, object]],
    *,
    model: str,
    allowed_tools: frozenset[str] | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    on_tool_call: ToolCallHook | None = None,
) -> LoopResult:
    """Run the model with `tools` until it answers or `max_rounds` is hit.

    `messages` is mutated in place (assistant + tool turns appended) so a
    caller can keep the transcript. `allowed_tools=None` exposes every
    tool; a concrete set scopes to a persona.
    """
    tools = registry.list_tools(allowed_tools)
    tool_calls_made: list[str] = []
    message: dict[str, Any] = {}
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None

    for _ in range(max_rounds):
        response = llm.chat_completion(model=model, messages=messages, tools=tools)
        message = cast(dict[str, Any], response["choices"][0]["message"])
        usage = cast(dict[str, Any] | None, response.get("usage"))
        cost_usd = cast(float | None, response.get("response_cost_usd"))
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

    return LoopResult(message=message, tool_calls=tool_calls_made, usage=usage, cost_usd=cost_usd)


def last_text(message: dict[str, Any]) -> str:
    """The assistant text from a loop result, or "" when it ended on a
    tool call with no content."""
    content = message.get("content")
    return content if isinstance(content, str) else ""
