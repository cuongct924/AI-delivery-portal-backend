"""services/orchestration-api/routers/chat.py — patches the module-level
adapter singletons, same pattern as tests/test_rag_router.py. `send_message`
is async; tests without tools pass a bare mock for `http_request`.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from persona_tool_scope import allowed_tools_for
from routers.chat import ChatRequest, send_message


def _http_request(mcp_registry: MagicMock | None = None) -> MagicMock:
    request = MagicMock()
    request.app.state.mcp_registry = mcp_registry or MagicMock()
    return request


@pytest.mark.asyncio
async def test_send_message_uses_active_prompt_and_forwards_model() -> None:
    request = ChatRequest(message="hi", persona="mlops", model="llama-3-8b-self-hosted")
    with (
        patch("routers.chat.registry_adapter") as mock_registry,
        patch("routers.chat.llm_gateway_adapter") as mock_gateway,
    ):
        mock_registry.get_active_version.return_value = "3"
        mock_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 42},
            "response_cost_usd": 0.001,
        }

        response = await send_message(request, _http_request())

    assert response.reply == "hello"
    assert response.persona_version == "3"
    assert response.rag_index_version is None
    assert response.tokens == 42
    assert response.cost_usd == pytest.approx(0.001)
    assert response.tools_used == []
    assert response.pending_confirmation is None
    mock_gateway.chat_completion.assert_called_once_with(
        model="llama-3-8b-self-hosted",
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
        ],
    )


@pytest.mark.asyncio
async def test_send_message_raises_404_when_persona_has_no_active_version() -> None:
    request = ChatRequest(message="hi", persona="unknown-persona")
    with patch("routers.chat.registry_adapter") as mock_registry:
        mock_registry.get_active_version.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await send_message(request, _http_request())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_send_message_use_rag_without_collection_raises_400() -> None:
    request = ChatRequest(message="hi", use_rag=True, rag_collection=None)
    with patch("routers.chat.registry_adapter") as mock_registry:
        mock_registry.get_active_version.return_value = "1"
        with pytest.raises(HTTPException) as exc_info:
            await send_message(request, _http_request())
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_send_message_use_rag_without_active_index_raises_400() -> None:
    request = ChatRequest(message="hi", use_rag=True, rag_collection="smoke-test")
    with patch("routers.chat.registry_adapter") as mock_registry:
        mock_registry.get_active_version.side_effect = ["1", None]
        with pytest.raises(HTTPException) as exc_info:
            await send_message(request, _http_request())
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_send_message_use_rag_prepends_context_and_returns_index_version() -> None:
    request = ChatRequest(message="hi", use_rag=True, rag_collection="smoke-test")
    with (
        patch("routers.chat.registry_adapter") as mock_registry,
        patch("routers.chat.llm_gateway_adapter") as mock_gateway,
        patch("routers.chat.vector_store_adapter") as mock_vector_store,
    ):
        mock_registry.get_active_version.side_effect = ["1", "2"]  # prompt, then rag-index
        mock_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.embed.return_value = [[0.1]]
        mock_vector_store.search.return_value = [{"payload": {"text": "retrieved chunk"}}]
        mock_gateway.chat_completion.return_value = {"choices": [{"message": {"content": "hello"}}]}

        response = await send_message(request, _http_request())

    assert response.rag_index_version == "2"
    mock_gateway.chat_completion.assert_called_once_with(
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": "Context:\n\nretrieved chunk\n\nsystem prompt"},
            {"role": "user", "content": "hi"},
        ],
    )


@pytest.mark.asyncio
async def test_send_message_use_tools_calls_auto_executable_tool() -> None:
    request = ChatRequest(message="draft a prompt", use_tools=True)
    mock_mcp_registry = MagicMock()
    mock_mcp_registry.list_tools.return_value = [
        {"type": "function", "function": {"name": "draft_prompt"}}
    ]
    mock_mcp_registry.is_destructive.return_value = False

    async def fake_call_tool(name: str, args: dict) -> str:
        assert name == "draft_prompt"
        assert args == {"name": "x"}
        return "tool result"

    mock_mcp_registry.call_tool = fake_call_tool

    tool_call_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "draft_prompt", "arguments": json.dumps({"name": "x"})},
            }
        ],
    }

    with (
        patch("routers.chat.registry_adapter") as mock_registry,
        patch("routers.chat.llm_gateway_adapter") as mock_gateway,
    ):
        mock_registry.get_active_version.return_value = "1"
        mock_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.side_effect = [
            {"choices": [{"message": tool_call_message}], "usage": {"total_tokens": 10}},
            {"choices": [{"message": {"content": "done"}}], "usage": {"total_tokens": 20}},
        ]

        response = await send_message(request, _http_request(mock_mcp_registry))

    assert response.reply == "done"
    assert response.tools_used == ["draft_prompt"]
    assert response.pending_confirmation is None
    assert mock_gateway.chat_completion.call_count == 2
    second_call_messages = mock_gateway.chat_completion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool result",
    }


@pytest.mark.asyncio
async def test_send_message_use_tools_gates_destructive_tool_behind_confirmation() -> None:
    """Destructive tools must never be auto-executed."""
    request = ChatRequest(message="activate version 2", use_tools=True)
    mock_mcp_registry = MagicMock()
    mock_mcp_registry.list_tools.return_value = [
        {"type": "function", "function": {"name": "activate_prompt"}}
    ]
    mock_mcp_registry.is_destructive.return_value = True
    mock_mcp_registry.call_tool = MagicMock(side_effect=AssertionError("must not be called"))

    tool_call_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "activate_prompt",
                    "arguments": json.dumps({"name": "mlops", "version": "2"}),
                },
            }
        ],
    }

    with (
        patch("routers.chat.registry_adapter") as mock_registry,
        patch("routers.chat.llm_gateway_adapter") as mock_gateway,
    ):
        mock_registry.get_active_version.return_value = "1"
        mock_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.return_value = {
            "choices": [{"message": tool_call_message}],
            "usage": {"total_tokens": 10},
        }

        response = await send_message(request, _http_request(mock_mcp_registry))

    assert response.tools_used == []
    assert response.pending_confirmation is not None
    assert "activate_prompt" in response.pending_confirmation
    assert mock_gateway.chat_completion.call_count == 1  # no follow-up call
    assert response.pending_tool_call == {
        "name": "activate_prompt",
        "arguments": {"name": "mlops", "version": "2"},
    }


@pytest.mark.asyncio
async def test_send_message_use_tools_scopes_tool_list_to_persona() -> None:
    """An unregistered persona is deny-by-default (empty tool set) — the
    model must never even be offered a mutating tool for it."""
    request = ChatRequest(message="how's the cluster", use_tools=True, persona="unregistered")
    mock_mcp_registry = MagicMock()
    mock_mcp_registry.list_tools.return_value = []

    with (
        patch("routers.chat.registry_adapter") as mock_registry,
        patch("routers.chat.llm_gateway_adapter") as mock_gateway,
    ):
        mock_registry.get_active_version.return_value = "1"
        mock_registry.get_version.return_value = {"content": "system prompt"}
        mock_gateway.chat_completion.return_value = {
            "choices": [{"message": {"content": "all good", "tool_calls": None}}]
        }

        await send_message(request, _http_request(mock_mcp_registry))

    mock_mcp_registry.list_tools.assert_called_once_with(allowed_tools_for("unregistered"))
    assert "activate_prompt" not in allowed_tools_for("unregistered")


@pytest.mark.asyncio
async def test_send_message_confirmed_tool_call_executes_directly_without_the_llm() -> None:
    """The resubmission path after a human approves pending_tool_call — never
    re-asks the model, and always forces confirm=True itself."""
    request = ChatRequest(
        message="yes, activate it",
        confirmed_tool_call={
            "name": "activate_prompt",
            "arguments": {"name": "mlops", "version": "2"},
        },
    )
    mock_mcp_registry = MagicMock()

    async def fake_call_tool(name: str, args: dict) -> str:
        assert name == "activate_prompt"
        assert args == {"name": "mlops", "version": "2", "confirm": True}
        return "activated"

    mock_mcp_registry.call_tool = fake_call_tool

    with (
        patch("routers.chat.registry_adapter") as mock_registry,
        patch("routers.chat.llm_gateway_adapter") as mock_gateway,
    ):
        mock_registry.get_active_version.return_value = "1"
        mock_registry.get_version.return_value = {"content": "system prompt"}

        response = await send_message(request, _http_request(mock_mcp_registry))

    assert response.reply == "activated"
    assert response.tools_used == ["activate_prompt"]
    mock_gateway.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_confirmed_tool_call_rejects_tool_outside_persona_scope() -> None:
    request = ChatRequest(
        message="do it",
        persona="unregistered",
        confirmed_tool_call={"name": "activate_prompt", "arguments": {}},
    )
    mock_mcp_registry = MagicMock()
    mock_mcp_registry.call_tool = MagicMock(side_effect=AssertionError("must not be called"))

    with patch("routers.chat.registry_adapter") as mock_registry:
        mock_registry.get_active_version.return_value = "1"
        mock_registry.get_version.return_value = {"content": "system prompt"}
        with pytest.raises(HTTPException) as exc_info:
            await send_message(request, _http_request(mock_mcp_registry))

    assert exc_info.value.status_code == 403
