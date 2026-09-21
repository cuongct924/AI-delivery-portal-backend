"""Tests adapters/ai_platform/mlflow_adapter.py.

Stubs the "mlflow" package at the sys.modules level before importing (same
pattern as tests/test_evaluate_drift.py — adapters/ai_platform/mlflow_adapter.py has
`import mlflow` at module level, and the real mlflow package is heavier than
a unit test needs), then patches `mlflow` and `MlflowClient` for the
duration of each test so assertions are made against plain `MagicMock`
objects rather than attributes typed as the real (unmocked) SDK classes —
pyright would otherwise resolve `adapter.client.<method>` to a real bound
method and reject `.assert_called_once_with(...)` on it.
"""

import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from adapters.ai_platform.mlflow_adapter import MlflowAdapter  # noqa: E402


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_mlflow() -> Iterator[MagicMock]:
    with patch("adapters.ai_platform.mlflow_adapter.mlflow") as mocked:
        yield mocked


@pytest.fixture
def adapter(mock_client: MagicMock, mock_mlflow: MagicMock) -> MlflowAdapter:
    with patch("adapters.ai_platform.mlflow_adapter.MlflowClient", return_value=mock_client):
        return MlflowAdapter(tracking_uri="http://mlflow.test")


def test_register_model_without_dataset_version_skips_tagging(
    adapter: MlflowAdapter, mock_client: MagicMock, mock_mlflow: MagicMock
) -> None:
    registered = MagicMock()
    registered.name = "fraud-model"
    registered.version = "3"
    mock_mlflow.register_model.return_value = registered

    result = adapter.register_model("fraud-model", "runs:/abc/model")

    assert result == {"name": "fraud-model", "version": "3"}
    mock_client.set_model_version_tag.assert_not_called()


def test_register_model_with_dataset_version_sets_tag(
    adapter: MlflowAdapter, mock_client: MagicMock, mock_mlflow: MagicMock
) -> None:
    registered = MagicMock()
    registered.name = "fraud-model"
    registered.version = "4"
    mock_mlflow.register_model.return_value = registered

    result = adapter.register_model("fraud-model", "runs:/abc/model", dataset_version="d41d8cd9")

    assert result == {"name": "fraud-model", "version": "4"}
    mock_client.set_model_version_tag.assert_called_once_with(
        "fraud-model", "4", "dataset_version", "d41d8cd9"
    )


def test_set_model_version_tag_wraps_client(adapter: MlflowAdapter, mock_client: MagicMock) -> None:
    adapter.set_model_version_tag("fraud-model", "2", "gate_passed", "true")

    mock_client.set_model_version_tag.assert_called_once_with(
        "fraud-model", "2", "gate_passed", "true"
    )


def test_get_model_version_details_combines_version_and_run(
    adapter: MlflowAdapter, mock_client: MagicMock
) -> None:
    mv = MagicMock()
    mv.version = "2"
    mv.run_id = "run-1"
    mv.tags = {"gate_passed": "true"}
    mv.status = "READY"
    mock_client.get_model_version.return_value = mv
    run = MagicMock()
    run.data.metrics = {"accuracy": 0.9}
    mock_client.get_run.return_value = run

    result = adapter.get_model_version_details("fraud-model", "2")

    assert result == {
        "version": "2",
        "run_id": "run-1",
        "tags": {"gate_passed": "true"},
        "metrics": {"accuracy": 0.9},
        "status": "READY",
    }


def test_get_model_version_details_raises_without_run_id(
    adapter: MlflowAdapter, mock_client: MagicMock
) -> None:
    mv = MagicMock()
    mv.run_id = None
    mock_client.get_model_version.return_value = mv

    with pytest.raises(ValueError, match="no associated run_id"):
        adapter.get_model_version_details("fraud-model", "2")


def test_get_latest_version_returns_highest_version(
    adapter: MlflowAdapter, mock_client: MagicMock
) -> None:
    def _mv(version: str) -> MagicMock:
        mock = MagicMock()
        mock.version = version
        return mock

    mock_client.search_model_versions.return_value = [_mv("1"), _mv("10"), _mv("2")]

    assert adapter.get_latest_version("fraud-model") == "10"


def test_get_latest_version_raises_when_no_versions(
    adapter: MlflowAdapter, mock_client: MagicMock
) -> None:
    mock_client.search_model_versions.return_value = []

    with pytest.raises(ValueError, match="no registered versions"):
        adapter.get_latest_version("fraud-model")
