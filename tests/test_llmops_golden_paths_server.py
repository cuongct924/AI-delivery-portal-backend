"""agents/mcp-servers/llmops-golden-paths-server/server.py's
_require_confirmed_mutation — the in-body gate for activate_prompt/
rag_activate, checked here directly rather than through the
@mcp.tool()-wrapped functions.

Loaded via conftest.load_mcp_server (not a bare `from server import ...`)
because mlops-golden-paths-server has its own same-named server.py/
token_verifier.py/thunder_client.py — see that helper's docstring.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import load_mcp_server

_SERVER_DIR = Path(__file__).parent.parent / "agents/mcp-servers/llmops-golden-paths-server"
llmops_server = load_mcp_server(_SERVER_DIR, "llmops_golden_paths")


def test_rejects_when_not_confirmed() -> None:
    error = llmops_server._require_confirmed_mutation(confirm=False)

    assert error is not None
    assert error["error"] == "confirmation required"


def test_rejects_confirmed_call_with_no_access_token() -> None:
    with patch("llmops_golden_paths_server.get_access_token", return_value=None):
        error = llmops_server._require_confirmed_mutation(confirm=True)

    assert error is not None
    assert llmops_server.MUTATE_SCOPE in error["error"]


def test_rejects_confirmed_call_with_token_missing_scope() -> None:
    token = MagicMock(scopes=["some-other-scope"])
    with patch("llmops_golden_paths_server.get_access_token", return_value=token):
        error = llmops_server._require_confirmed_mutation(confirm=True)

    assert error is not None
    assert llmops_server.MUTATE_SCOPE in error["error"]


def test_allows_confirmed_call_with_correct_scope() -> None:
    token = MagicMock(scopes=[llmops_server.MUTATE_SCOPE])
    with patch("llmops_golden_paths_server.get_access_token", return_value=token):
        error = llmops_server._require_confirmed_mutation(confirm=True)

    assert error is None


def test_mutate_scope_is_distinct_from_mlops_golden_paths_server() -> None:
    # Two different domains, two different scopes — a token authorized to
    # trigger training/deploy must not automatically satisfy this server's
    # (LLMOps prompt/RAG) gate, and vice versa.
    assert llmops_server.MUTATE_SCOPE == "llmops-golden-paths:mutate"
