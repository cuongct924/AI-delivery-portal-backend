"""agents/mcp-servers/mlops-golden-paths-server/server.py's
_require_confirmed_mutation — the in-body gate for trigger_training/
register_model/prepare_deploy/prepare_llm_deploy.

Loaded via conftest.load_mcp_server (not a bare `from server import ...`)
because llmops-golden-paths-server has its own same-named server.py/
token_verifier.py/thunder_client.py — see that helper's docstring.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import load_mcp_server

_SERVER_DIR = Path(__file__).parent.parent / "agents/mcp-servers/mlops-golden-paths-server"
mlops_server = load_mcp_server(_SERVER_DIR, "mlops_golden_paths")


def test_rejects_when_not_confirmed() -> None:
    error = mlops_server._require_confirmed_mutation(confirm=False)

    assert error is not None
    assert error["error"] == "confirmation required"


def test_rejects_confirmed_call_with_no_access_token() -> None:
    with patch("mlops_golden_paths_server.get_access_token", return_value=None):
        error = mlops_server._require_confirmed_mutation(confirm=True)

    assert error is not None
    assert mlops_server.MUTATE_SCOPE in error["error"]


def test_rejects_confirmed_call_with_token_missing_scope() -> None:
    token = MagicMock(scopes=["some-other-scope"])
    with patch("mlops_golden_paths_server.get_access_token", return_value=token):
        error = mlops_server._require_confirmed_mutation(confirm=True)

    assert error is not None
    assert mlops_server.MUTATE_SCOPE in error["error"]


def test_allows_confirmed_call_with_correct_scope() -> None:
    token = MagicMock(scopes=[mlops_server.MUTATE_SCOPE])
    with patch("mlops_golden_paths_server.get_access_token", return_value=token):
        error = mlops_server._require_confirmed_mutation(confirm=True)

    assert error is None


def test_mutate_scope_is_distinct_from_llmops_golden_paths_server() -> None:
    # Two different domains, two different scopes — a token authorized to
    # mutate LLMOps prompt/RAG state must not automatically satisfy this
    # server's gate.
    assert mlops_server.MUTATE_SCOPE == "mlops-golden-paths:mutate"


def test_validate_dataset_posts_to_datasets_validate_without_confirmation_gate() -> None:
    with patch("mlops_golden_paths_server._post", return_value=[]) as mock_post:
        mlops_server.validate_dataset(dataset_uri="file:///a.csv", task_type="classification")

    mock_post.assert_called_once_with(
        "/datasets/validate",
        {
            "dataset_uri": "file:///a.csv",
            "task_type": "classification",
            "target_column": None,
            "time_column": None,
        },
    )
