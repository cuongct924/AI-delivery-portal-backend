"""Implements the wire protocol OpenChoreo's Portal Assistant chat drawer
expects (see backstage-plugins' `plugins/openchoreo-portal-assistant{,-backend}`)
so that already-built UI (FAB, drawer, streaming rendering) works against
this repo's own MLOps/LLMOps chat + MCP tool-calling — instead of the
separate "perch-agent" service OpenChoreo itself doesn't ship in either
repo. Point the frontend's `OPENCHOREO_PORTAL_ASSISTANT_URL` at this
orchestration-api's base URL and the drawer talks to this router directly.

Read-only by design, mirroring the frontend forwarder's own comment ("the
agent is read-only — there is no /execute route to proxy"): a destructive
tool (activate_prompt/rag_activate) is described in the reply, never
called — routers/chat.py's `pending_confirmation` flow is the one place
that owns executing those, and this endpoint doesn't replicate it.

`scope` (Component/Environment/etc. from the frontend's OpenChoreo pages)
is accepted but unused — none of its fields map to this repo's domain
(personas, models, prompts) yet.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, Final, cast

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from adapters.factory import get_llm_gateway_adapter, get_registry_adapter

router = APIRouter(prefix="/api/v1alpha1/portal-assistant", tags=["portal-assistant"])

llm_gateway_adapter = get_llm_gateway_adapter()
registry_adapter = get_registry_adapter()

PERSONA: Final[str] = "mlops"
MODEL: Final[str] = "claude-sonnet-5"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatScope(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PortalAssistantChatRequest(BaseModel):
    messages: list[ChatMessage]
    scope: ChatScope | None = None


def _sse_line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event) + "\n").encode()


async def _stream_reply(
    request: Request, chat_request: PortalAssistantChatRequest
) -> AsyncIterator[bytes]:
    try:
        active_version = registry_adapter.get_active_version("prompt", PERSONA)
        system_prompt = (
            registry_adapter.get_version("prompt", PERSONA, active_version)["content"]
            if active_version is not None
            else "You are the MLOps assistant for the AI Delivery Portal."
        )

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            *[{"role": m.role, "content": m.content} for m in chat_request.messages],
        ]

        registry = request.app.state.mcp_registry
        response = llm_gateway_adapter.chat_completion(
            model=MODEL, messages=messages, tools=registry.list_tools()
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            call = tool_calls[0]  # bounded to 1 tool call per turn, same as routers/chat.py
            tool_name = call["function"]["name"]
            tool_args = call["function"]["arguments"]

            if registry.is_destructive(tool_name):
                done_message = (
                    f"I'd need to call the tool '{tool_name}' with {tool_args} to do "
                    "that, but this assistant is read-only and can't execute changes — "
                    "please use the corresponding Golden Path template instead."
                )
                yield _sse_line({"type": "done", "message": done_message})
                return

            yield _sse_line({"type": "tool_call", "tool": tool_name, "args": tool_args})
            tool_result = await registry.call_tool(tool_name, json.loads(tool_args))
            # cast: message is a TypedDict (ChatCompletionMessage), not
            # assignable to dict[str, object] by static invariance rules,
            # even though it's a plain dict at runtime — same as routers/chat.py.
            messages.append(cast(dict[str, object], message))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result})
            response = llm_gateway_adapter.chat_completion(model=MODEL, messages=messages)
            message = response["choices"][0]["message"]

        reply = message["content"] or ""
        yield _sse_line({"type": "message_chunk", "content": reply})
        yield _sse_line({"type": "done", "message": reply})
    except Exception as exc:  # noqa: BLE001 - surfaced to the drawer as a StreamEvent, not a 500
        yield _sse_line({"type": "error", "message": str(exc)})


@router.post("/chat")
async def portal_assistant_chat(
    chat_request: PortalAssistantChatRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    del user  # auth-gated only, not read inside the stream
    return StreamingResponse(
        _stream_reply(request, chat_request), media_type="application/x-ndjson"
    )


@router.post("/warmup")
def portal_assistant_warmup(user: dict = Depends(get_current_user)) -> dict[str, str]:
    # The MCP registry already connects at app startup (main.py's lifespan)
    # — nothing per-user to pre-warm here, unlike perch-agent's own cache.
    del user
    return {"status": "ok"}
