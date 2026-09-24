"""Implements OpenChoreo Portal Assistant's chat wire protocol
(AI-delivery-portal-frontend's `plugins/openchoreo-portal-assistant{,-backend}`) so
its existing drawer UI works against this repo's own MLOps/LLMOps chat +
MCP tools instead of OpenChoreo's own perch-agent service.

Read-only by design: a destructive tool is described in the reply, never
called — routers/chat.py's `pending_confirmation` flow owns executing
those.

Beyond plain chat, this surface lets the agent fill a Golden Path
template: it calls `list_golden_paths` / `get_golden_path_schema` /
`propose_golden_path_draft` (golden-path-guide-server MCP), and each
`propose_golden_path_draft` result is merged into the session's draft and
streamed to the drawer as a `template_draft` event. The drawer seeds the
Scaffolder form with it; the user reviews and submits — the agent never
submits. Message history + draft are persisted per `session_id` so a
reload restores both.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Final

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from tool_loop import last_text, run_tool_loop

from adapters.factory import (
    get_chat_session_store,
    get_llm_gateway_adapter,
    get_prompt_registry_adapter,
)

router = APIRouter(prefix="/api/v1alpha1/portal-assistant", tags=["portal-assistant"])

llm_gateway_adapter = get_llm_gateway_adapter()
# Prompts live in MLflow's Prompt Registry (see factory.py).
registry_adapter = get_prompt_registry_adapter()
session_store = get_chat_session_store()

PERSONA: Final[str] = "mlops"
MODEL: Final[str] = "claude-sonnet-5"
# The tool whose result becomes a `template_draft` event.
DRAFT_TOOL: Final[str] = "propose_golden_path_draft"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatScope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Hint: the template the user currently has open, so the agent can skip
    # the list_golden_paths round. Never the source of truth — the agent
    # still reads the live schema.
    currentTemplate: str | None = None


class PortalAssistantChatRequest(BaseModel):
    messages: list[ChatMessage]
    scope: ChatScope | None = None
    session_id: str | None = None


def _sse_line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event) + "\n").encode()


def _user_ref(user: dict) -> str:
    return str(user.get("sub") or user.get("preferred_username") or "unknown")


async def _stream_reply(
    request: Request, chat_request: PortalAssistantChatRequest, user_ref: str
) -> AsyncIterator[bytes]:
    try:
        active_version = registry_adapter.get_active_version("prompt", PERSONA)
        system_prompt = (
            str(registry_adapter.get_version("prompt", PERSONA, active_version)["content"])
            if active_version is not None
            else "You are the MLOps assistant for the AI Delivery Portal."
        )
        if chat_request.scope and chat_request.scope.currentTemplate:
            system_prompt += (
                f"\n\nThe user currently has the Golden Path template "
                f"'{chat_request.scope.currentTemplate}' open."
            )

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            *[{"role": m.role, "content": m.content} for m in chat_request.messages],
        ]

        registry = request.app.state.mcp_registry
        session_id = chat_request.session_id
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def on_tool_call(name: str, args: dict[str, Any], result: str) -> None:
            await queue.put({"type": "tool_call", "tool": name, "args": json.dumps(args)})
            if name != DRAFT_TOOL or session_id is None:
                return
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                return
            template = str(args.get("name", ""))
            form_data = payload.get("form_data", {})
            if not template or not isinstance(form_data, dict):
                return
            merged = session_store.merge_draft(session_id, user_ref, template, form_data)
            await queue.put(
                {
                    "type": "template_draft",
                    "template": template,
                    "formData": merged["form_data"],
                    "missing": payload.get("missing", []),
                    "complete": bool(payload.get("ok")),
                }
            )

        async def run() -> None:
            try:
                result = await run_tool_loop(
                    llm_gateway_adapter,
                    registry,
                    messages,
                    model=MODEL,
                    on_tool_call=on_tool_call,
                )
                if result.pending is not None:
                    reply = (
                        f"I'd need to call the tool '{result.pending['name']}' with "
                        f"{result.pending['arguments']} to do that, but this assistant "
                        "is read-only and can't execute changes — please use the "
                        "corresponding Golden Path template instead."
                    )
                else:
                    reply = last_text(result.message)
                if session_id is not None:
                    turns = [{"role": m.role, "content": m.content} for m in chat_request.messages]
                    turns.append({"role": "assistant", "content": reply})
                    session_store.append_messages(session_id, user_ref, turns)
                await queue.put({"type": "message_chunk", "content": reply})
                await queue.put({"type": "done", "message": reply})
            except Exception as exc:  # noqa: BLE001 - surfaced as a StreamEvent
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield _sse_line(event)
        await task
    except Exception as exc:  # noqa: BLE001 - surfaced to the drawer as a StreamEvent, not a 500
        yield _sse_line({"type": "error", "message": str(exc)})


@router.post("/chat")
async def portal_assistant_chat(
    chat_request: PortalAssistantChatRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_reply(request, chat_request, _user_ref(user)),
        media_type="application/x-ndjson",
    )


@router.get("/sessions/{session_id}")
def get_session(session_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Restore a session's message history + draft after a reload. Scoped to
    the owner — a guessed id can't read someone else's draft."""
    session = session_store.get(session_id)
    if session is None or session["user_ref"] != _user_ref(user):
        return {"messages": [], "draft": None}
    return {"messages": session["messages"], "draft": session["draft"]}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(get_current_user)) -> dict[str, bool]:
    del user
    session_store.clear(session_id)
    return {"deleted": True}


@router.post("/warmup")
def portal_assistant_warmup(user: dict = Depends(get_current_user)) -> dict[str, str]:
    # The MCP registry already connects at app startup (main.py's lifespan)
    # — nothing per-user to pre-warm here, unlike perch-agent's own cache.
    del user
    return {"status": "ok"}
