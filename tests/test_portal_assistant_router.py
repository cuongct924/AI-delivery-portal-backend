"""services/orchestration-api/routers/portal_assistant.py — patches the
module-level adapter singletons, same pattern as tests/test_chat_router.py.
`_stream_reply` is an async generator; tests collect and decode its
yielded NDJSON lines.

`llm_gateway_adapter.chat_completion_stream` (not `chat_completion`) is
what `run_tool_loop` calls now — mocked via `conftest.make_stream`, which
turns a list of delta-chunk dicts per round into a side_effect-ready async
generator factory. See that helper's docstring for why a plain
`.side_effect = [list, of, dicts]` (the old `chat_completion` pattern)
doesn't work for a streaming method.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import make_stream
from core.config import settings
from routers.portal_assistant import (
    ChatMessage,
    PortalAssistantChatRequest,
    _stream_reply,
    portal_assistant_warmup,
)


def _http_request(mcp_registry: MagicMock | None = None) -> MagicMock:
    request = MagicMock()
    request.app.state.mcp_registry = mcp_registry or MagicMock()
    return request


async def _collect(chat_request: PortalAssistantChatRequest, request: MagicMock) -> list[dict]:
    return [json.loads(line) async for line in _stream_reply(request, chat_request, "dev")]


@pytest.mark.asyncio
async def test_reply_without_tool_call_emits_message_chunk_then_done() -> None:
    chat_request = PortalAssistantChatRequest(messages=[ChatMessage(role="user", content="hi")])
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = []
    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = make_stream(
            [{"choices": [{"delta": {"content": "hello"}}]}]
        )

        events = await _collect(chat_request, _http_request(mock_registry))

    # "hello" streams as a single delta here, landing as one message_chunk —
    # a real reply would arrive as several smaller fragments, but the wire
    # shape (one event per delta, no final duplicate blob) is what's under
    # test, not fragment granularity.
    assert events == [
        {"type": "message_chunk", "content": "hello"},
        {"type": "done", "message": "hello"},
    ]
    mock_gateway.chat_completion_stream.assert_called_once_with(
        # Not a hardcoded literal — settings.portal_assistant_model is
        # env-configurable (PORTAL_ASSISTANT_MODEL in .env, exported by the
        # Makefile into every recipe including `make test`), and a
        # hardcoded "claude-sonnet-5" here would fail for anyone who's
        # overridden it locally.
        model=settings.portal_assistant_model,
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
        ],
        tools=[],
    )


@pytest.mark.asyncio
async def test_falls_back_to_default_prompt_when_no_active_version() -> None:
    chat_request = PortalAssistantChatRequest(messages=[ChatMessage(role="user", content="hi")])
    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = None
        mock_gateway.chat_completion_stream.side_effect = make_stream(
            [{"choices": [{"delta": {"content": "hello"}}]}]
        )

        await _collect(chat_request, _http_request())

    system_message = mock_gateway.chat_completion_stream.call_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "MLOps assistant" in system_message["content"]


@pytest.mark.asyncio
async def test_non_destructive_tool_call_emits_tool_call_event_then_final_reply() -> None:
    chat_request = PortalAssistantChatRequest(
        messages=[ChatMessage(role="user", content="list experiments")]
    )
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = [{"type": "function", "function": {"name": "t"}}]
    mock_registry.is_destructive.return_value = False
    mock_registry.call_tool = AsyncMock(return_value='{"name": "fraud-detection"}')

    first_round = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "list_experiments", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    ]
    second_round = [{"choices": [{"delta": {"content": "There is 1 experiment."}}]}]

    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = make_stream(first_round, second_round)

        events = await _collect(chat_request, _http_request(mock_registry))

    assert events == [
        {"type": "tool_call", "tool": "list_experiments", "args": "{}"},
        {"type": "message_chunk", "content": "There is 1 experiment."},
        {"type": "done", "message": "There is 1 experiment."},
    ]
    mock_registry.call_tool.assert_awaited_once_with("list_experiments", {})
    assert mock_gateway.chat_completion_stream.call_count == 2


@pytest.mark.asyncio
async def test_propose_draft_emits_template_draft_event() -> None:
    chat_request = PortalAssistantChatRequest(
        messages=[ChatMessage(role="user", content="create a fraud model")],
        session_id="s1",
    )
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = [{"type": "function", "function": {"name": "t"}}]
    mock_registry.is_destructive.return_value = False
    mock_registry.call_tool = AsyncMock(
        return_value=json.dumps(
            {"ok": True, "form_data": {"modelName": "fraud"}, "missing": [], "errors": []}
        )
    )

    first_round = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "propose_golden_path_draft",
                                    "arguments": json.dumps(
                                        {
                                            "name": "train-track-register",
                                            "values": {"modelName": "fraud"},
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
    ]
    second_round = [{"choices": [{"delta": {"content": "Filled it in."}}]}]

    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
        patch("routers.portal_assistant.session_store") as mock_store,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = make_stream(first_round, second_round)

        events = await _collect(chat_request, _http_request(mock_registry))

    # formData is the tool's own raw result now — nothing merges it with a
    # prior turn's fields on this side any more (that moved to the Node
    # DraftService, frontend repo).
    draft_events = [event for event in events if event["type"] == "template_draft"]
    assert draft_events == [
        {
            "type": "template_draft",
            "template": "train-track-register",
            "formData": {"modelName": "fraud"},
            "missing": [],
            "complete": True,
        }
    ]
    mock_store.append_messages.assert_called_once()


def test_chat_request_accepts_camelcase_session_id() -> None:
    # The frontend's ChatRequest serialises the session id as camelCase
    # `sessionId` (matching scope.currentTemplate). The agent must accept
    # it — otherwise session_id is None, on_tool_call returns early, and no
    # template_draft event is ever emitted.
    request = PortalAssistantChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "sessionId": "s1"}
    )
    assert request.session_id == "s1"


def test_chat_request_accepts_snakecase_session_id() -> None:
    request = PortalAssistantChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "session_id": "s1"}
    )
    assert request.session_id == "s1"


@pytest.mark.asyncio
async def test_reasoning_deltas_stream_as_reasoning_chunk_events() -> None:
    chat_request = PortalAssistantChatRequest(messages=[ChatMessage(role="user", content="hi")])
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = []
    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = make_stream(
            [
                {"choices": [{"delta": {"reasoning_content": "Let me"}}]},
                {"choices": [{"delta": {"reasoning_content": " think."}}]},
                {"choices": [{"delta": {"content": "hello"}}]},
            ]
        )

        events = await _collect(chat_request, _http_request(mock_registry))

    # Reasoning arrives as its own event type, distinct from — and ahead
    # of — the visible reply, so the drawer can show live "thinking"
    # progress instead of a static indicator for however long the model
    # takes before it starts producing the actual answer.
    assert events == [
        {"type": "reasoning_chunk", "content": "Let me"},
        {"type": "reasoning_chunk", "content": " think."},
        {"type": "message_chunk", "content": "hello"},
        {"type": "done", "message": "hello"},
    ]


@pytest.mark.asyncio
async def test_multi_chunk_tool_call_arguments_are_concatenated_by_index() -> None:
    """A long propose_golden_path_draft call can have its `arguments`
    string split across several deltas (confirmed live against the real
    gateway for a many-field `values` dict) — this must reassemble to
    valid JSON before it's parsed."""
    chat_request = PortalAssistantChatRequest(
        messages=[ChatMessage(role="user", content="fill the form")]
    )
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = [{"type": "function", "function": {"name": "t"}}]
    mock_registry.is_destructive.return_value = False
    mock_registry.call_tool = AsyncMock(return_value="{}")

    full_args = json.dumps({"name": "train-track-register", "values": {"modelName": "fraud"}})
    midpoint = len(full_args) // 2
    first_round = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "propose_golden_path_draft"},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": full_args[:midpoint]}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": full_args[midpoint:]}}
                        ]
                    }
                }
            ]
        },
    ]
    second_round = [{"choices": [{"delta": {"content": "done"}}]}]

    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = make_stream(first_round, second_round)

        events = await _collect(chat_request, _http_request(mock_registry))

    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert tool_call_events == [
        {
            "type": "tool_call",
            "tool": "propose_golden_path_draft",
            "args": full_args,
            "activeForm": "Đang điền form…",
        }
    ]


@pytest.mark.asyncio
async def test_destructive_tool_call_is_refused_without_executing() -> None:
    chat_request = PortalAssistantChatRequest(
        messages=[ChatMessage(role="user", content="activate the new prompt")]
    )
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = [{"type": "function", "function": {"name": "t"}}]
    mock_registry.is_destructive.return_value = True
    mock_registry.call_tool = AsyncMock()

    first_round = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "activate_prompt",
                                    "arguments": '{"name": "mlops", "version": "2"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    ]

    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = make_stream(first_round)

        events = await _collect(chat_request, _http_request(mock_registry))

    # The refusal text is synthesized after the loop ends (never streamed
    # as deltas), so it still needs its own message_chunk — same shape as
    # before streaming existed.
    assert [event["type"] for event in events] == ["message_chunk", "done"]
    assert "read-only" in events[-1]["message"]
    mock_registry.call_tool.assert_not_awaited()
    mock_gateway.chat_completion_stream.assert_called_once()


@pytest.mark.asyncio
async def test_llm_gateway_error_emits_error_event_instead_of_raising() -> None:
    chat_request = PortalAssistantChatRequest(messages=[ChatMessage(role="user", content="hi")])
    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion_stream.side_effect = RuntimeError("upstream boom")

        events = await _collect(chat_request, _http_request())

    assert events == [{"type": "error", "message": "upstream boom"}]


def test_warmup_returns_ok() -> None:
    assert portal_assistant_warmup(user={}) == {"status": "ok"}
