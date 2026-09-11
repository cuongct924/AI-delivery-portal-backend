"""services/orchestration-api/routers/portal_assistant.py — patches the
module-level adapter singletons, same pattern as tests/test_chat_router.py.
`_stream_reply` is an async generator; tests collect and decode its
yielded NDJSON lines.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
    return [json.loads(line) async for line in _stream_reply(request, chat_request)]


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
        mock_gateway.chat_completion.return_value = {"choices": [{"message": {"content": "hello"}}]}

        events = await _collect(chat_request, _http_request(mock_registry))

    assert events == [
        {"type": "message_chunk", "content": "hello"},
        {"type": "done", "message": "hello"},
    ]
    mock_gateway.chat_completion.assert_called_once_with(
        model="claude-sonnet-5",
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
        mock_gateway.chat_completion.return_value = {"choices": [{"message": {"content": "hello"}}]}

        await _collect(chat_request, _http_request())

    system_message = mock_gateway.chat_completion.call_args.kwargs["messages"][0]
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

    first_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "list_experiments", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    second_response = {"choices": [{"message": {"content": "There is 1 experiment."}}]}

    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.side_effect = [first_response, second_response]

        events = await _collect(chat_request, _http_request(mock_registry))

    assert events == [
        {"type": "tool_call", "tool": "list_experiments", "args": "{}"},
        {"type": "message_chunk", "content": "There is 1 experiment."},
        {"type": "done", "message": "There is 1 experiment."},
    ]
    mock_registry.call_tool.assert_awaited_once_with("list_experiments", {})
    assert mock_gateway.chat_completion.call_count == 2


@pytest.mark.asyncio
async def test_destructive_tool_call_is_refused_without_executing() -> None:
    chat_request = PortalAssistantChatRequest(
        messages=[ChatMessage(role="user", content="activate the new prompt")]
    )
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = [{"type": "function", "function": {"name": "t"}}]
    mock_registry.is_destructive.return_value = True
    mock_registry.call_tool = AsyncMock()

    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "activate_prompt",
                                "arguments": '{"name": "mlops", "version": "2"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.return_value = response

        events = await _collect(chat_request, _http_request(mock_registry))

    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert "read-only" in events[0]["message"]
    mock_registry.call_tool.assert_not_awaited()
    mock_gateway.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_llm_gateway_error_emits_error_event_instead_of_raising() -> None:
    chat_request = PortalAssistantChatRequest(messages=[ChatMessage(role="user", content="hi")])
    with (
        patch("routers.portal_assistant.registry_adapter") as mock_prompt_registry,
        patch("routers.portal_assistant.llm_gateway_adapter") as mock_gateway,
    ):
        mock_prompt_registry.get_active_version.return_value = "1"
        mock_prompt_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.side_effect = RuntimeError("upstream boom")

        events = await _collect(chat_request, _http_request())

    assert events == [{"type": "error", "message": "upstream boom"}]


def test_warmup_returns_ok() -> None:
    assert portal_assistant_warmup(user={}) == {"status": "ok"}
