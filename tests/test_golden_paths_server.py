"""agents/mcp-servers/golden-paths-server/server.py's _require_confirmed_mutation
— the in-body gate for activate_prompt/rag_activate, checked here directly
rather than through the @mcp.tool()-wrapped functions."""

from unittest.mock import MagicMock, patch

from server import MUTATE_SCOPE, _require_confirmed_mutation


def test_rejects_when_not_confirmed() -> None:
    error = _require_confirmed_mutation(confirm=False)

    assert error is not None
    assert error["error"] == "confirmation required"


def test_rejects_confirmed_call_with_no_access_token() -> None:
    with patch("server.get_access_token", return_value=None):
        error = _require_confirmed_mutation(confirm=True)

    assert error is not None
    assert MUTATE_SCOPE in error["error"]


def test_rejects_confirmed_call_with_token_missing_scope() -> None:
    token = MagicMock(scopes=["some-other-scope"])
    with patch("server.get_access_token", return_value=token):
        error = _require_confirmed_mutation(confirm=True)

    assert error is not None
    assert MUTATE_SCOPE in error["error"]


def test_allows_confirmed_call_with_correct_scope() -> None:
    token = MagicMock(scopes=[MUTATE_SCOPE])
    with patch("server.get_access_token", return_value=token):
        error = _require_confirmed_mutation(confirm=True)

    assert error is None
