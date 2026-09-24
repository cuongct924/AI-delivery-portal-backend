"""services/orchestration-api/catalog_client.py — patches `httpx.get` and
`settings` directly, same pattern as tests/test_auth_thunder.py."""

from unittest.mock import MagicMock, patch

from catalog_client import (
    discover_mcp_servers,
    get_golden_path_schema,
    get_golden_path_template,
    list_golden_path_templates,
)


def test_discover_mcp_servers_parses_matching_entities() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "metadata": {
                "name": "llmops-golden-paths-mcp",
                "annotations": {
                    "mcp/endpoint": "http://llmops-golden-paths-server:9002/mcp",
                    "mcp/transport": "streamable-http",
                },
            }
        },
        {
            "metadata": {
                "name": "ai-observability-mcp",
                "annotations": {
                    "mcp/endpoint": "http://ai-observability-server:9001/mcp",
                    "mcp/transport": "streamable-http",
                },
            }
        },
    ]
    with patch("catalog_client.httpx.get", return_value=mock_response):
        servers = discover_mcp_servers()

    assert servers == [
        {
            "name": "llmops-golden-paths-mcp",
            "endpoint": "http://llmops-golden-paths-server:9002/mcp",
            "transport": "streamable-http",
        },
        {
            "name": "ai-observability-mcp",
            "endpoint": "http://ai-observability-server:9001/mcp",
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


def test_list_golden_path_templates_filters_by_tag() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "metadata": {
                "name": "train-track-register",
                "title": "Train & Register Model",
                "description": "Trains a model.\n",
                "tags": ["mlops"],
            }
        },
        {
            "metadata": {
                "name": "create-openchoreo-clustertrait",
                "title": "ClusterTrait",
                "tags": ["platform"],
            }
        },
    ]
    with patch("catalog_client.httpx.get", return_value=mock_response):
        templates = list_golden_path_templates()

    assert templates == [
        {
            "name": "train-track-register",
            "title": "Train & Register Model",
            "description": "Trains a model.",
            "tags": ["mlops"],
        }
    ]


def test_list_golden_path_templates_returns_empty_list_on_request_failure() -> None:
    with patch("catalog_client.httpx.get", side_effect=Exception("connection refused")):
        assert list_golden_path_templates() == []


def test_get_golden_path_template_extracts_parameters_and_steps() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": {
            "name": "train-track-register",
            "title": "Train & Register Model",
            "description": "Trains a model.\n",
            "tags": ["mlops"],
        },
        "spec": {
            "parameters": [
                {
                    "required": ["modelName"],
                    "properties": {
                        "modelName": {"title": "Model name", "description": "E.g. foo\n"},
                        "notes": {"title": "Notes"},
                    },
                }
            ],
            "steps": [{"id": "train", "name": "Train", "action": "orchestration:trigger-training"}],
        },
    }
    with patch("catalog_client.httpx.get", return_value=mock_response):
        template = get_golden_path_template("train-track-register")

    assert template == {
        "name": "train-track-register",
        "title": "Train & Register Model",
        "description": "Trains a model.",
        "tags": ["mlops"],
        "parameters": [
            {
                "name": "modelName",
                "title": "Model name",
                "description": "E.g. foo",
                "required": True,
            },
            {"name": "notes", "title": "Notes", "description": "", "required": False},
        ],
        "steps": [{"name": "Train", "action": "orchestration:trigger-training"}],
    }


def test_get_golden_path_schema_returns_raw_parameters() -> None:
    raw = [
        {
            "required": ["modelName"],
            "properties": {"modelName": {"type": "string", "default": "foo"}},
            "allOf": [{"if": {"properties": {"x": {"const": "y"}}}, "then": {"required": ["z"]}}],
        }
    ]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": {"name": "train-track-register", "tags": ["mlops"]},
        "spec": {"parameters": raw},
    }
    with patch("catalog_client.httpx.get", return_value=mock_response):
        schema = get_golden_path_schema("train-track-register")

    assert schema == raw


def test_get_golden_path_schema_returns_none_for_wrong_tag() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": {"name": "create-openchoreo-clustertrait", "tags": ["platform"]},
        "spec": {"parameters": []},
    }
    with patch("catalog_client.httpx.get", return_value=mock_response):
        assert get_golden_path_schema("create-openchoreo-clustertrait") is None


def test_get_golden_path_template_returns_none_for_wrong_tag() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": {"name": "create-openchoreo-clustertrait", "tags": ["platform"]},
        "spec": {},
    }
    with patch("catalog_client.httpx.get", return_value=mock_response):
        assert get_golden_path_template("create-openchoreo-clustertrait") is None


def test_get_golden_path_template_returns_none_on_404() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 404
    with patch("catalog_client.httpx.get", return_value=mock_response):
        assert get_golden_path_template("does-not-exist") is None
