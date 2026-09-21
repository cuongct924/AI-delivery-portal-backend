"""adapters/ai_platform/huggingface_hub_adapter.py — patches `httpx.Client` so no real
network call is made, same patch-at-the-boundary style as
tests/test_auth_thunder.py's `_jwks`/`jwt.decode` patches.
"""

from unittest.mock import MagicMock, patch

from adapters.ai_platform.huggingface_hub_adapter import HuggingFaceHubAdapter


def _response(status_code: int, json_body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body or {}
    return response


def _mock_client(responses: dict[str, MagicMock]) -> MagicMock:
    """`responses` maps a URL suffix to the MagicMock response `client.get()`
    should return for it."""
    client = MagicMock()

    def get(url: str, *args: object, **kwargs: object) -> MagicMock:
        for suffix, response in responses.items():
            if url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected URL: {url}")

    client.get.side_effect = get
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    return client


def test_get_model_info_not_found_returns_exists_false() -> None:
    client = _mock_client({"/api/models/nonexistent/model": _response(404)})
    with patch("adapters.ai_platform.huggingface_hub_adapter.httpx.Client", return_value=client):
        info = HuggingFaceHubAdapter().get_model_info("nonexistent/model")

    assert info["exists"] is False
    assert info["is_gated"] is False
    assert info["num_layers"] is None


def test_get_model_info_gated_without_access_returns_is_gated_true() -> None:
    client = _mock_client({"/api/models/meta-llama/Llama-3.1-8B-Instruct": _response(401)})
    with patch("adapters.ai_platform.huggingface_hub_adapter.httpx.Client", return_value=client):
        info = HuggingFaceHubAdapter().get_model_info("meta-llama/Llama-3.1-8B-Instruct")

    assert info["exists"] is True
    assert info["is_gated"] is True
    assert info["num_layers"] is None


def test_get_model_info_public_model_returns_full_architecture() -> None:
    model_response = _response(
        200,
        {
            "gated": False,
            "safetensors": {"total": 7_000_000_000},
            "cardData": {"license": "apache-2.0"},
        },
    )
    config_response = _response(
        200,
        {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
        },
    )
    client = _mock_client(
        {
            "/api/models/mistralai/Mistral-7B-Instruct-v0.3": model_response,
            "/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/config.json": config_response,
        }
    )
    with patch("adapters.ai_platform.huggingface_hub_adapter.httpx.Client", return_value=client):
        info = HuggingFaceHubAdapter().get_model_info("mistralai/Mistral-7B-Instruct-v0.3")

    assert info["exists"] is True
    assert info["is_gated"] is False
    assert info["param_count_billion"] == 7.0
    assert info["num_layers"] == 32
    assert info["max_context_length"] == 32768
    assert info["license"] == "apache-2.0"


def test_get_model_info_gated_with_token_reads_config() -> None:
    model_response = _response(200, {"gated": "manual", "safetensors": {"total": 8_030_000_000}})
    config_response = _response(
        200,
        {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
        },
    )
    client = _mock_client(
        {
            "/api/models/meta-llama/Llama-3.1-8B-Instruct": model_response,
            "/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json": config_response,
        }
    )
    with (
        patch("adapters.ai_platform.huggingface_hub_adapter.httpx.Client", return_value=client),
        patch("adapters.ai_platform.huggingface_hub_adapter.settings") as mock_settings,
    ):
        mock_settings.huggingface_hub_token = "hf_faketoken"
        info = HuggingFaceHubAdapter().get_model_info("meta-llama/Llama-3.1-8B-Instruct")

    assert info["is_gated"] is True
    assert info["num_layers"] == 32
