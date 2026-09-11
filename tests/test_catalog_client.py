"""services/orchestration-api/catalog_client.py — patches `httpx.get` and
`settings` directly, same pattern as tests/test_auth_thunder.py."""

from unittest.mock import MagicMock, patch

from catalog_client import discover_mcp_servers


def test_discover_mcp_servers_parses_matching_entities() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "metadata": {
                "name": "golden-paths-mcp",
                "annotations": {
                    "mcp/endpoint": "http://golden-paths-server:9002/mcp",
                    "mcp/transport": "streamable-http",
                },
            }
        },
        {
            "metadata": {
                "name": "observability-mcp",
                "annotations": {
                    "mcp/endpoint": "http://observability-server:9001/mcp",
                    "mcp/transport": "streamable-http",
                },
            }
        },
    ]
    with patch("catalog_client.httpx.get", return_value=mock_response):
        servers = discover_mcp_servers()

    assert servers == [
        {
            "name": "golden-paths-mcp",
            "endpoint": "http://golden-paths-server:9002/mcp",
            "transport": "streamable-http",
        },
        {
            "name": "observability-mcp",
            "endpoint": "http://observability-server:9001/mcp",
            "transport": "streamable-http",
        },
    ]


def test_discover_mcp_servers_skips_entities_missing_annotations() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"metadata": {"name": "incomplete-mcp", "annotations": {}}},
    ]
    with patch("catalog_client.httpx.get", return_value=mock_response):
        servers = discover_mcp_servers()
    assert servers == []


def test_discover_mcp_servers_returns_empty_list_on_request_failure() -> None:
    """Backstage/Catalog being unreachable must not crash orchestration-api
    startup — tool-calling just becomes unavailable."""
    with patch("catalog_client.httpx.get", side_effect=Exception("connection refused")):
        servers = discover_mcp_servers()
    assert servers == []


def test_discover_mcp_servers_sends_bearer_token_when_configured() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = []
    with (
        patch("catalog_client.httpx.get", return_value=mock_response) as mock_get,
        patch("catalog_client.settings") as mock_settings,
    ):
        mock_settings.backstage_base_url = "http://localhost:7007"
        mock_settings.backstage_service_token = "a-real-token"
        discover_mcp_servers()

    _, kwargs = mock_get.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer a-real-token"}
