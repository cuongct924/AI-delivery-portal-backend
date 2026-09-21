"""Tests adapters/ai_platform/mock_model_registry_adapter.py."""

import pytest

from adapters.ai_platform.mock_model_registry_adapter import MockModelRegistryAdapter


@pytest.fixture
def adapter() -> MockModelRegistryAdapter:
    return MockModelRegistryAdapter()


def test_register_model_auto_increments_version_per_name(
    adapter: MockModelRegistryAdapter,
) -> None:
    first = adapter.register_model("fraud-detection", "runs:/abc/model")
    second = adapter.register_model("fraud-detection", "runs:/def/model")

    assert first == {"name": "fraud-detection", "version": "1"}
    assert second == {"name": "fraud-detection", "version": "2"}


def test_register_model_with_dataset_version_tags_it(adapter: MockModelRegistryAdapter) -> None:
    adapter.register_model("fraud-detection", "runs:/abc/model", dataset_version="d41d8cd9")

    details = adapter.get_model_version_details("fraud-detection", "1")
    assert details["tags"]["dataset_version"] == "d41d8cd9"


def test_registered_metrics_pass_every_task_type_threshold(
    adapter: MockModelRegistryAdapter,
) -> None:
    from evaluations.evaluate_gate import TASK_TYPE_THRESHOLDS, evaluate_metrics_gate

    adapter.register_model("fraud-detection", "runs:/abc/model")
    metrics = adapter.get_model_metrics("fraud-detection", "1")

    for task_type in TASK_TYPE_THRESHOLDS:
        assert evaluate_metrics_gate(task_type, metrics)["passed"] is True


def test_set_model_version_tag_then_get_details_reflects_it(
    adapter: MockModelRegistryAdapter,
) -> None:
    adapter.register_model("fraud-detection", "runs:/abc/model")

    adapter.set_model_version_tag("fraud-detection", "1", "task_type", "classification")

    details = adapter.get_model_version_details("fraud-detection", "1")
    assert details["tags"]["task_type"] == "classification"


def test_get_model_version_details_raises_for_unregistered_version(
    adapter: MockModelRegistryAdapter,
) -> None:
    with pytest.raises(ValueError, match="not registered"):
        adapter.get_model_version_details("fraud-detection", "1")


def test_list_models_returns_every_registered_name(adapter: MockModelRegistryAdapter) -> None:
    adapter.register_model("fraud-detection", "runs:/abc/model")
    adapter.register_model("churn-prediction", "runs:/def/model")

    names = {m["name"] for m in adapter.list_models()}
    assert names == {"fraud-detection", "churn-prediction"}


def test_get_latest_version_returns_highest_version(adapter: MockModelRegistryAdapter) -> None:
    adapter.register_model("fraud-detection", "runs:/abc/model")
    adapter.register_model("fraud-detection", "runs:/def/model")

    assert adapter.get_latest_version("fraud-detection") == "2"


def test_get_latest_version_raises_when_no_versions(adapter: MockModelRegistryAdapter) -> None:
    with pytest.raises(ValueError, match="no registered versions"):
        adapter.get_latest_version("fraud-detection")


def test_get_dataset_lineage_uses_tagged_dataset_version(
    adapter: MockModelRegistryAdapter,
) -> None:
    adapter.register_model("fraud-detection", "runs:/abc/model", dataset_version="d41d8cd9")

    lineage = adapter.get_dataset_lineage("fraud-detection", "1")

    assert lineage == [
        {
            "name": "fraud-detection-training-data",
            "digest": "d41d8cd9",
            "source": "mock://datasets/fraud-detection",
        }
    ]
