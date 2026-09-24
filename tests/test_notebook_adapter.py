"""adapters/ai_platform/notebook_adapter.py — patches httpx, never touches a
real JupyterHub."""

from unittest.mock import MagicMock, patch

import pytest

from adapters.ai_platform.notebook_adapter import JupyterHubAdapter


@pytest.fixture(autouse=True)
def _clear_jupyterhub_env(monkeypatch) -> None:
    # The ambient shell may have .env sourced — pin the defaults the tests
    # assert on instead of depending on it.
    monkeypatch.delenv("JUPYTERHUB_URL", raising=False)
    monkeypatch.delenv("JUPYTERHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("JUPYTERHUB_PUBLIC_URL", raising=False)


def _adapter() -> JupyterHubAdapter:
    return JupyterHubAdapter(base_url="http://hub.test", token="tok")


def _response(status: int = 200, json: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = json if json is not None else {}
    response.raise_for_status = MagicMock()
    return response


def test_create_notebook_creates_the_user_before_spawning() -> None:
    # JupyterHub's spawn endpoint 404s for a user that doesn't exist yet.
    with patch("adapters.ai_platform.notebook_adapter.httpx") as mock_httpx:
        mock_httpx.post.side_effect = [
            _response(201),  # POST /hub/api/users
            _response(201),  # POST /hub/api/users/{id}/server
        ]
        mock_httpx.get.return_value = _response(200, {"server": "/user/nb-x/"})
        _adapter().create_notebook("sklearn-cpu", 8)

    create_user = mock_httpx.post.call_args_list[0]
    assert create_user.args[0] == "http://hub.test/hub/api/users"
    assert create_user.kwargs["json"] == {"usernames": [create_user.kwargs["json"]["usernames"][0]]}


def test_spawn_sends_flat_user_options_not_a_nested_dict() -> None:
    # KubeSpawner reads user_options as flat keys; a nested `profile_options`
    # dict is silently ignored and every option falls back to its default.
    with patch("adapters.ai_platform.notebook_adapter.httpx") as mock_httpx:
        mock_httpx.post.side_effect = [_response(201), _response(201)]
        mock_httpx.get.return_value = _response(200, {"server": None})
        _adapter().create_notebook("sklearn-cpu", 8, gpu_type=None)

    spawn = mock_httpx.post.call_args_list[1]
    body = spawn.kwargs["json"]
    assert "profile_options" not in body
    assert body["profile"] == "ai-notebook"
    assert body["environment"] == "sklearn-cpu"
    assert body["gpu_type"] == "none"
    # Numeric options are stringified to match the spawner's choice keys.
    assert body["cpu_cores"] == "2"
    assert body["gpu_count"] == "0"


def test_get_notebook_status_returns_an_absolute_url() -> None:
    with patch("adapters.ai_platform.notebook_adapter.httpx") as mock_httpx:
        mock_httpx.get.return_value = _response(200, {"server": "/user/nb-x/"})
        status = _adapter().get_notebook_status("nb-x")

    assert status["url"] == "http://localhost:8888/user/nb-x/"
    assert status["active"] is True


def test_create_notebook_returns_a_url_even_before_the_server_is_ready() -> None:
    # create_notebook returns right after spawn; the server takes ~20s to
    # start, so `server` is still null. The URL must not be empty or the
    # Portal's "Open the notebook" link renders blank.
    with patch("adapters.ai_platform.notebook_adapter.httpx") as mock_httpx:
        mock_httpx.post.side_effect = [_response(201), _response(201)]
        mock_httpx.get.return_value = _response(200, {"server": None})
        status = _adapter().create_notebook("sklearn-cpu", 8)

    assert status["url"] == f"http://localhost:8888/user/{status['notebook_id']}/"
    assert status["active"] is False


def test_get_notebook_status_raises_value_error_on_404() -> None:
    with patch("adapters.ai_platform.notebook_adapter.httpx") as mock_httpx:
        mock_httpx.get.return_value = _response(404)
        with pytest.raises(ValueError, match="not found"):
            _adapter().get_notebook_status("nb-missing")


def test_empty_token_omits_the_authorization_header(monkeypatch) -> None:
    # An empty token produced "Illegal header value b'token '" and 500'd.
    monkeypatch.delenv("JUPYTERHUB_API_TOKEN", raising=False)
    adapter = JupyterHubAdapter(base_url="http://hub.test", token="")
    assert adapter._headers == {}
