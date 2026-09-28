"""Implements OpenChoreo Portal Assistant's chat wire protocol
(AI-delivery-portal-frontend's `plugins/openchoreo-portal-assistant{,-backend}`) so
its existing drawer UI works against this repo's own MLOps/LLMOps chat +
MCP tools instead of OpenChoreo's own perch-agent service.

Read-only by design: destructive tools (trigger_training, register_model,
prepare_deploy, prepare_llm_deploy, activate_prompt, rag_activate) aren't
even offered to the model here (persona_tool_scope.PORTAL_ASSISTANT_TOOLS) —
routers/chat.py's `pending_confirmation` flow owns executing those, on a
separate chat surface.

Beyond plain chat, this surface lets the agent fill a Golden Path
template: it calls `list_golden_paths` / `get_golden_path_schema` /
`propose_golden_path_draft` (golden-path-guide-server MCP), and each
`propose_golden_path_draft` result is streamed to the drawer as-is, as a
`template_draft` event. The drawer seeds the Scaffolder form with it; the
user reviews and submits — the agent never submits.

For a business-defining field with a fixed set of options, the model
calls the synthetic `ask_user` tool (ask_user_tool.py — not a real MCP
tool, never dispatched through the registry) instead of asking in prose;
its arguments become a `clarifying_question` event so the drawer can
render clickable chips instead of text the user has to retype.

Only message history is persisted per `session_id` here (so a reload
restores the conversation) — the draft itself is NOT tracked on this side
any more. It used to be (merged turn-by-turn via
adapters.ai_platform.chat_session_store's old `merge_draft`), but that was
a second, independently-updated copy of state the frontend's
portal-assistant-backend Node plugin already owned more completely
(draftId/revision/run-status, none of which existed here) — two stores
drifting out of sync was the actual bug, not a redundancy worth keeping.
Each `template_draft` event below now carries only the current turn's raw
tool result; accumulating fields across turns is the Node
DraftService.upsertFromEvent's job now, not this router's.

`tool_loop.run_tool_loop`'s `on_delta` hook streams the model's own
reasoning/content tokens as they're produced (a reasoning model like the
one behind PORTAL_ASSISTANT_MODEL can take 15-45s across several tool
rounds; without this the drawer had nothing to show but a static
"Thinking…" the whole time) — reasoning becomes a `reasoning_chunk` event,
content becomes incremental `message_chunk` events instead of one blob at
the very end.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Final, Literal

from ask_user_tool import ASK_USER_TOOL, ASK_USER_TOOL_NAME
from auth.thunder import get_current_user
from core.config import settings
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from mcp_client import ToolSchema
from persona_tool_scope import PORTAL_ASSISTANT_TOOLS
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from tool_loop import last_text, run_tool_loop
from ttl_cache import TTLCache

from adapters.factory import (
    get_chat_session_store,
    get_llm_gateway_adapter,
    get_prompt_registry_adapter,
)
from routers.golden_paths import summarize_golden_path_schema

router = APIRouter(prefix="/api/v1alpha1/portal-assistant", tags=["portal-assistant"])

llm_gateway_adapter = get_llm_gateway_adapter()
# Prompts live in MLflow's Prompt Registry (see factory.py).
registry_adapter = get_prompt_registry_adapter()
session_store = get_chat_session_store()

PERSONA: Final[str] = "mlops"
# The tool whose result becomes a `template_draft` event.
DRAFT_TOOL: Final[str] = "propose_golden_path_draft"
# Offered to the model alongside the real MCP tools; a call to it stops the
# loop as `LoopResult.local_call` instead of being dispatched — see
# ask_user_tool.py.
LOCAL_TOOLS: Final[dict[str, ToolSchema]] = {ASK_USER_TOOL_NAME: ASK_USER_TOOL}
# 10 exchanges (user+assistant pairs) — generous for a Golden Path fill,
# which typically resolves in 2-4 turns; only kicks in for an unusually
# long conversation.
_MAX_HISTORY_MESSAGES: Final[int] = 20

# Short TTL: batches the repeated Prompt Registry / Catalog reads within one
# conversation's burst of messages, without meaningfully delaying testing a
# newly `/prompts/{name}/activate`d prompt (see ttl_cache.py's docstring).
_CACHE_TTL_SECONDS: Final[float] = 30.0
_system_prompt_cache: TTLCache[str] = TTLCache(_CACHE_TTL_SECONDS)
_schema_summary_cache: TTLCache[str] = TTLCache(_CACHE_TTL_SECONDS)


def _fetch_system_prompt() -> str:
    active_version = registry_adapter.get_active_version("prompt", PERSONA)
    return (
        str(registry_adapter.get_version("prompt", PERSONA, active_version)["content"])
        if active_version is not None
        else "You are the MLOps assistant for the AI Delivery Portal."
    )


_FOLLOWUP_SYSTEM_PROMPT: Final[str] = (
    "Suggest 2-3 short, natural follow-up questions a user might ask next, "
    "given the most recent reply in a support chat. Return ONLY a JSON "
    "array of strings — no prose, no markdown fencing, nothing else. Each "
    "question under 60 characters. If nothing reasonable comes to mind, "
    "return an empty array []."
)


def _fetch_followup_suggestions(reply: str) -> list[str]:
    """A separate, single-purpose call — not embedded as a trailer in the
    main reply — so the primary generation stays focused purely on
    answering well, and this stays trivially parseable regardless of how
    long or complex that reply is. Best-effort: any failure (call error,
    malformed JSON, wrong shape) just means no suggestions, never fails
    the turn — same advisory-only contract as every other secondary
    signal in this router."""
    try:
        response = llm_gateway_adapter.chat_completion(
            model=settings.portal_assistant_model,
            messages=[
                {"role": "system", "content": _FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Assistant's reply:\n{reply}"},
            ],
        )
        content = response["choices"][0]["message"]["content"] or "[]"
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            return []
        return [str(item)[:100] for item in parsed if isinstance(item, str)][:3]
    except Exception:  # noqa: BLE001 - advisory only, never fails the turn
        return []


# Human-readable progress label per tool, streamed as a `tool_call` event's
# `activeForm`. The drawer falls back to "Running <tool>…" for anything not
# listed, so a new tool needs no change here to keep working.
TOOL_ACTIVE_FORM: Final[dict[str, str]] = {
    "list_golden_paths": "Đang tìm quy trình phù hợp…",
    "get_golden_path_guide": "Đang đọc hướng dẫn quy trình…",
    "get_golden_path_schema": "Đang đọc cấu hình biểu mẫu…",
    "propose_golden_path_draft": "Đang điền form…",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatScope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Hint: the template the user currently has open, so the agent can skip
    # the list_golden_paths round. Never the source of truth — the agent
    # still reads the live schema.
    currentTemplate: str | None = None
    # The Scaffolder form's OWN current field values for currentTemplate —
    # not just what this agent last proposed itself. The frontend only
    # sets this when its shared draft store's template matches
    # currentTemplate, so it's never stale data from a different template.
    # Lets the agent treat a field the user edited directly on the form
    # (never mentioned in chat) as already-settled instead of re-asking or
    # silently overwriting it.
    currentFormValues: dict[str, Any] | None = None


class PortalAssistantChatRequest(BaseModel):
    messages: list[ChatMessage]
    scope: ChatScope | None = None
    # The frontend's ChatRequest serialises this as camelCase `sessionId`
    # (matching `scope.currentTemplate`); accept the snake_case form too so
    # both wire shapes work. Without this the agent saw session_id=None and
    # skipped the `template_draft` event entirely (on_tool_call returns
    # early when there's no session_id to scope the event to).
    session_id: str | None = Field(
        default=None, validation_alias=AliasChoices("sessionId", "session_id")
    )


def _sse_line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event) + "\n").encode()


def _user_ref(user: dict) -> str:
    return str(user.get("sub") or user.get("preferred_username") or "unknown")


async def _stream_reply(
    request: Request, chat_request: PortalAssistantChatRequest, user_ref: str
) -> AsyncIterator[bytes]:
    try:
        system_prompt = _system_prompt_cache.get_or_set(PERSONA, _fetch_system_prompt)
        if chat_request.scope and chat_request.scope.currentTemplate:
            template = chat_request.scope.currentTemplate
            # Fetch the schema ourselves instead of letting the model spend
            # a full reasoning round deciding to call list_golden_paths/
            # get_golden_path_schema for a template we already know by name
            # — each round trip is model-latency-bound (10-20s+ for a
            # reasoning model), while this catalog lookup is a single sub-
            # second HTTP call. Best-effort: a catalog hiccup or unknown
            # name just falls back to the pre-existing behavior (the model
            # looks it up itself via MCP).
            #
            # Condensed rendering (summarize_golden_path_schema), not the
            # raw JSON Schema — same fields/types/branching, ~58% fewer
            # tokens (measured on train-track-register). `/draft` still
            # validates the full schema; only what's embedded here changed.
            summary = _schema_summary_cache.get(template)
            if summary is None:
                summary = await asyncio.to_thread(summarize_golden_path_schema, template)
                if summary is not None:
                    _schema_summary_cache.set(template, summary)
            if summary is not None:
                system_prompt += (
                    f"\n\nThe user currently has the Golden Path template "
                    f"'{template}' open. Its fields:\n"
                    f"{summary}\n"
                    "Do NOT call list_golden_paths or get_golden_path_schema for "
                    f"this template — you already have it. Go straight to "
                    "propose_golden_path_draft."
                )
            else:
                system_prompt += (
                    f"\n\nThe user currently has the Golden Path template '{template}' open."
                )
            if chat_request.scope.currentFormValues:
                system_prompt += (
                    "\n\nThe form currently has these values — some may be the "
                    "user's own manual edits made directly on the form, not "
                    "through this chat:\n"
                    f"{json.dumps(chat_request.scope.currentFormValues)}\n"
                    "Treat every field listed here as already settled — don't "
                    "ask about it again or silently overwrite it with a "
                    "different value unless the user's own message says "
                    "otherwise."
                )

        # Cap what's sent to the model — the frontend resends the FULL
        # conversation on every turn (see AssistantChatDrawer.tsx's
        # messageHistory), so an unbounded history means unbounded
        # per-turn cost/latency as a conversation runs long. Only trims
        # what goes to the LLM; session_store below still persists every
        # turn for reload/audit. Dropped, not summarized — simpler, and
        # this assistant's own state (the draft) lives in
        # templateDraftStore/the server-side draft record, not in
        # conversation memory, so losing older turns rarely loses
        # anything that actually matters to the current request.
        history = chat_request.messages[-_MAX_HISTORY_MESSAGES:]
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            *[{"role": m.role, "content": m.content} for m in history],
        ]

        registry = request.app.state.mcp_registry
        session_id = chat_request.session_id
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def on_tool_call(name: str, args: dict[str, Any], result: str) -> None:
            event: dict[str, Any] = {"type": "tool_call", "tool": name, "args": json.dumps(args)}
            if name in TOOL_ACTIVE_FORM:
                event["activeForm"] = TOOL_ACTIVE_FORM[name]
            await queue.put(event)
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
            await queue.put(
                {
                    "type": "template_draft",
                    "template": template,
                    "formData": form_data,
                    "missing": payload.get("missing", []),
                    "complete": bool(payload.get("ok")),
                }
            )

        async def on_delta(kind: Literal["reasoning", "content"], text: str) -> None:
            event_type = "reasoning_chunk" if kind == "reasoning" else "message_chunk"
            await queue.put({"type": event_type, "content": text})

        async def run() -> None:
            try:
                result = await run_tool_loop(
                    llm_gateway_adapter,
                    registry,
                    messages,
                    model=settings.portal_assistant_model,
                    allowed_tools=PORTAL_ASSISTANT_TOOLS,
                    local_tools=LOCAL_TOOLS,
                    on_tool_call=on_tool_call,
                    on_delta=on_delta,
                )
                # `done.message` is what the drawer falls back to rendering
                # as a plain assistant bubble — left empty when a dedicated
                # event (clarifying_question) already carried the content,
                # so the turn doesn't render twice.
                done_message = ""
                if (
                    result.local_call is not None
                    and result.local_call["name"] == ASK_USER_TOOL_NAME
                ):
                    question = str(result.local_call["arguments"].get("question", ""))
                    options = result.local_call["arguments"].get("options", [])
                    await queue.put(
                        {"type": "clarifying_question", "question": question, "options": options}
                    )
                    # Persisted as the assistant's turn so the next round's
                    # history still makes sense — a reload without the
                    # chip UI just shows this as a plain-text question,
                    # same as any other assistant message.
                    reply = question
                elif result.pending is not None:
                    reply = (
                        f"I'd need to call the tool '{result.pending['name']}' with "
                        f"{result.pending['arguments']} to do that, but this assistant "
                        "is read-only and can't execute changes — please use the "
                        "corresponding Golden Path template instead."
                    )
                    # Synthesized here, after the loop ended — never streamed
                    # as deltas, so (unlike the normal reply) this needs an
                    # explicit message_chunk or the drawer shows nothing.
                    await queue.put({"type": "message_chunk", "content": reply})
                    done_message = reply
                elif result.exhausted:
                    # The model was still mid-flow (calling more tools)
                    # when the round budget ran out — last_text(result.message)
                    # would be "" here (the last round never produced a
                    # final answer), which used to mean the drawer showed
                    # nothing with no explanation at all.
                    reply = (
                        "I made some progress but ran out of steps before "
                        "finishing this turn. Could you try again — maybe "
                        "with a more specific request, or answering any "
                        "questions I'd already asked?"
                    )
                    await queue.put({"type": "message_chunk", "content": reply})
                    done_message = reply
                else:
                    # Already streamed incrementally via on_delta above — a
                    # second, full-blob message_chunk here would just
                    # duplicate everything the drawer already rendered.
                    reply = last_text(result.message)
                    done_message = reply
                if session_id is not None:
                    turns = [{"role": m.role, "content": m.content} for m in chat_request.messages]
                    turns.append({"role": "assistant", "content": reply})
                    session_store.append_messages(session_id, user_ref, turns)
                done_event: dict[str, Any] = {"type": "done", "message": done_message}
                # Only for a normal answered turn — a clarifying_question is
                # itself the next interaction (nothing to suggest on top of
                # it), and a pending-tool refusal's own text already tells
                # the user what to do instead.
                if result.local_call is None and result.pending is None and reply:
                    suggestions = await asyncio.to_thread(_fetch_followup_suggestions, reply)
                    if suggestions:
                        done_event["suggestions"] = suggestions
                await queue.put(done_event)
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
    """Restore a session's message history after a reload. Scoped to the
    owner — a guessed id can't read someone else's messages.

    No draft in the response any more — that's the portal-assistant-backend
    Node plugin's `GET /drafts/by-session/{id}` now (frontend repo), which
    the drawer already calls separately and treats as authoritative. This
    endpoint used to also re-derive `missing`/`complete` against the live
    schema and return a `draft` key; both are gone since there's no draft
    state left here to derive them from."""
    session = session_store.get(session_id)
    if session is None or session["user_ref"] != _user_ref(user):
        return {"messages": []}
    return {"messages": session["messages"]}


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
