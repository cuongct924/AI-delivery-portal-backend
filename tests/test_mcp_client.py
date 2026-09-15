"""services/orchestration-api/mcp_client.py — patches `discover_mcp_servers`,
`streamable_http_client`, and `ClientSession` directly."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent, Tool, ToolAnnotations
from mcp_client import McpToolRegistry


def _tool(name: str, destructive: bool = False) -> Tool:
    return Tool(
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=destructive),
    )


def _fake_session(tools: list[Tool]) -> AsyncMock:
    session = AsyncMock()
    session.initialize = AsyncMock(return_value=None)
    session.list_tools = AsyncMock(return_value=MagicMock(tools=tools))
    return session


@pytest.mark.asyncio
async def test_connect_all_aggregates_tools_across_servers() -> None:
    server_a = {"name": "server-a", "endpoint": "http://a/mcp", "transport": "streamable-http"}
    server_b = {"name": "server-b", "endpoint": "http://b/mcp", "transport": "streamable-http"}

    session_a = _fake_session([_tool("tool_a")])
    session_b = _fake_session([_tool("tool_b", destructive=True)])
    session_by_url = {"http://a/mcp": session_a, "http://b/mcp": session_b}
    url_by_read_id: dict[int, str] = {}

    @asynccontextmanager
    async def fake_transport(url: str, **_kwargs: object):
        read, write = MagicMock(), MagicMock()
        url_by_read_id[id(read)] = url
        yield (read, write)

    @asynccontextmanager
    async def fake_client_session(read: object, write: object):
        yield session_by_url[url_by_read_id[id(read)]]

    with (
        patch("mcp_client.discover_mcp_servers", return_value=[server_a, server_b]),
        patch("mcp_client.mcp_auth_client.auth_headers", return_value={}),
        patch("mcp_client.streamable_http_client", side_effect=fake_transport),
        patch("mcp_client.ClientSession", side_effect=fake_client_session),
    ):
        registry = McpToolRegistry()
        await registry.connect_all()
        tool_names = {t["function"]["name"] for t in registry.list_tools()}
        assert tool_names == {"tool_a", "tool_b"}
        assert registry.is_destructive("tool_b") is True
        assert registry.is_destructive("tool_a") is False
        await registry.aclose()


@pytest.mark.asyncio
async def test_is_destructive_reflects_tool_annotation() -> None:
    registry = McpToolRegistry()
    registry._tools["activate_prompt"] = _tool("activate_prompt", destructive=True)
    registry._tools["draft_prompt"] = _tool("draft_prompt", destructive=False)

    assert registry.is_destructive("activate_prompt") is True
    assert registry.is_destructive("draft_prompt") is False
    assert registry.is_destructive("unknown_tool") is False


@pytest.mark.asyncio
async def test_list_tools_returns_openai_style_schema() -> None:
    registry = McpToolRegistry()
    registry._tools["draft_prompt"] = _tool("draft_prompt")

    schemas = registry.list_tools()

    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "draft_prompt",
                "description": "draft_prompt description",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_call_tool_raises_for_unknown_tool() -> None:
    registry = McpToolRegistry()
    with pytest.raises(ValueError, match="Unknown tool"):
        await registry.call_tool("does_not_exist", {})


@pytest.mark.asyncio
async def test_call_tool_extracts_text_content() -> None:
    registry = McpToolRegistry()
    session = AsyncMock()
    session.call_tool = AsyncMock(
        return_value=MagicMock(content=[TextContent(type="text", text="hello")])
    )
    registry._sessions["some-server"] = session
    registry._tool_to_server["draft_prompt"] = "some-server"

    result = await registry.call_tool("draft_prompt", {"name": "x"})

    assert result == "hello"
    session.call_tool.assert_awaited_once_with("draft_prompt", {"name": "x"})


@pytest.mark.asyncio
async def test_connect_all_isolates_one_failing_server() -> None:
    """A failed server must not block another server's tools."""
    good_server = {"name": "good", "endpoint": "http://good/mcp", "transport": "streamable-http"}
    bad_server = {"name": "bad", "endpoint": "http://bad/mcp", "transport": "streamable-http"}

    async def fake_run_server(self: McpToolRegistry, server: dict, ready) -> None:  # type: ignore[no-untyped-def]
        if server["name"] == "good":
            self._sessions["good"] = AsyncMock()
            self._tools["tool_ok"] = _tool("tool_ok")
            self._tool_to_server["tool_ok"] = "good"
        else:
            raise ConnectionError("simulated failure")
        ready.set()

    async def fake_run_server_wrapper(self, server, ready):  # type: ignore[no-untyped-def]
        try:
            await fake_run_server(self, server, ready)
        except Exception:
            ready.set()

    with (
        patch("mcp_client.discover_mcp_servers", return_value=[good_server, bad_server]),
        patch.object(McpToolRegistry, "_run_server", fake_run_server_wrapper),
    ):
        registry = McpToolRegistry()
        await registry.connect_all()

    assert "tool_ok" in registry.list_tools()[0]["function"]["name"]
    assert len(registry.list_tools()) == 1
