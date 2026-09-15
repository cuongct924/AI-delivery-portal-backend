"""Chat API — sends a message to the persona's active prompt via the LLM
Gateway, optionally with RAG context or MCP tool-calling.

use_tools=True lets the Agent call Golden Path tools, bounded to one tool
call per turn, filtered to the persona's allowed set (persona_tool_scope.py).
Destructive tools (activate_prompt, rag_activate) are only proposed via
ChatResponse.pending_tool_call, never auto-executed — the caller resubmits
it verbatim as ChatRequest.confirmed_tool_call only after a human approves,
which is the only path that ever executes one.
"""

import json
from typing import Final, cast

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request
from persona_tool_scope import allowed_tools_for
from pydantic import BaseModel

from adapters.factory import get_llm_gateway_adapter, get_registry_adapter, get_vector_store_adapter

router = APIRouter(prefix="/chat", tags=["chat"])

llm_gateway_adapter = get_llm_gateway_adapter()
vector_store_adapter = get_vector_store_adapter()
registry_adapter = get_registry_adapter()

EMBEDDING_MODEL: Final[str] = "voyage-3"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    persona: str = "mlops"
    use_rag: bool = False
    rag_collection: str | None = None
    # Lets the Agent call MCP tools during this turn.
    use_tools: bool = False
    # Overridable so chat isn't locked to Claude.
    model: str = "claude-sonnet-5"
    # Echo of a prior turn's ChatResponse.pending_tool_call, sent only after
    # the human explicitly approved it — {"name": ..., "arguments": {...}}.
    # This is the ONLY path that executes a destructive tool; chat.py (never
    # the model, never raw request input) is what adds "confirm": True.
    confirmed_tool_call: dict[str, object] | None = None


class ChatResponse(BaseModel):
    reply: str
    persona_version: str
    rag_index_version: str | None = None
    # None when the model has no cost entry in litellm-config.yaml (e.g. a
    # self-hosted model via the Serving LLM Golden Path).
    tokens: int
    cost_usd: float | None
    # Tools actually executed this turn.
    tools_used: list[str] = []
    # Set when a destructive tool was proposed but not executed.
    pending_confirmation: str | None = None
    # Structured form of the above — echo this back as
    # ChatRequest.confirmed_tool_call, unmodified, to actually run it.
    pending_tool_call: dict[str, object] | None = None


@router.post("", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest, http_request: Request, user: dict = Depends(get_current_user)
) -> ChatResponse:
    active_version = registry_adapter.get_active_version("prompt", chat_request.persona)
    if active_version is None:
        raise HTTPException(404, f"no active prompt version for persona {chat_request.persona!r}")
    system_prompt = registry_adapter.get_version("prompt", chat_request.persona, active_version)[
        "content"
    ]

    rag_version: str | None = None
    if chat_request.use_rag:
        if chat_request.rag_collection is None:
            raise HTTPException(400, "rag_collection is required when use_rag=True")
        rag_version = registry_adapter.get_active_version("rag-index", chat_request.rag_collection)
        if rag_version is None:
            raise HTTPException(
                400, f"no active RAG index for collection {chat_request.rag_collection!r}"
            )
        query_vector = llm_gateway_adapter.embed(EMBEDDING_MODEL, [chat_request.message])[0]
        hits = vector_store_adapter.search(query_vector, collection=chat_request.rag_collection)
        context = "\n\n".join(str(hit["payload"]["text"]) for hit in hits)
        system_prompt = f"Context:\n\n{context}\n\n{system_prompt}"

    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chat_request.message},
    ]

    tools_used: list[str] = []
    pending_confirmation: str | None = None

    if chat_request.confirmed_tool_call is not None:
        registry = http_request.app.state.mcp_registry
        tool_name = str(chat_request.confirmed_tool_call["name"])
        if tool_name not in allowed_tools_for(chat_request.persona):
            raise HTTPException(
                403, f"persona {chat_request.persona!r} is not allowed to call {tool_name!r}"
            )
        raw_arguments = chat_request.confirmed_tool_call.get("arguments")
        tool_args: dict[str, object] = (
            dict(raw_arguments) if isinstance(raw_arguments, dict) else {}
        )
        tool_result = await registry.call_tool(tool_name, {**tool_args, "confirm": True})
        return ChatResponse(
            reply=tool_result,
            persona_version=active_version,
            rag_index_version=rag_version,
            tokens=0,
            cost_usd=None,
            tools_used=[tool_name],
        )

    if chat_request.use_tools:
        registry = http_request.app.state.mcp_registry
        response = llm_gateway_adapter.chat_completion(
            model=chat_request.model,
            messages=messages,
            tools=registry.list_tools(allowed_tools_for(chat_request.persona)),
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            call = tool_calls[0]  # bounded to 1 tool call per turn
            tool_name = call["function"]["name"]
            tool_args = json.loads(call["function"]["arguments"])
            # Never let the model itself smuggle a mutation past the
            # confirmation gate — only the confirmed_tool_call branch above
            # is allowed to set this.
            tool_args.pop("confirm", None)

            if registry.is_destructive(tool_name):
                pending_confirmation = (
                    f"I'd like to call '{tool_name}' with {tool_args} — this "
                    "changes live state and needs your explicit confirmation "
                    "before I execute it."
                )
                pending_usage = response.get("usage")
                return ChatResponse(
                    reply=pending_confirmation,
                    persona_version=active_version,
                    rag_index_version=rag_version,
                    tokens=pending_usage["total_tokens"] if pending_usage is not None else 0,
                    cost_usd=response.get("response_cost_usd"),
                    tools_used=[],
                    pending_confirmation=pending_confirmation,
                    pending_tool_call={"name": tool_name, "arguments": tool_args},
                )

            tool_result = await registry.call_tool(tool_name, tool_args)
            tools_used.append(tool_name)
            # cast: message is a TypedDict (ChatCompletionMessage), not
            # assignable to dict[str, object] by static invariance rules,
            # even though it's a plain dict at runtime.
            messages.append(cast(dict[str, object], message))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result})
            response = llm_gateway_adapter.chat_completion(
                model=chat_request.model, messages=messages
            )
    else:
        response = llm_gateway_adapter.chat_completion(model=chat_request.model, messages=messages)

    reply = response["choices"][0]["message"]["content"] or ""
    usage = response.get("usage")
    tokens = usage["total_tokens"] if usage is not None else 0
    return ChatResponse(
        reply=reply,
        persona_version=active_version,
        rag_index_version=rag_version,
        tokens=tokens,
        cost_usd=response.get("response_cost_usd"),
        tools_used=tools_used,
        pending_confirmation=pending_confirmation,
    )
