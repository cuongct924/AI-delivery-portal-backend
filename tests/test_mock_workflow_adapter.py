"""Tests adapters/mock_workflow_adapter.py.

Return values are typed `dict[str, object]` (they pass through the Argo
Server API shape, per IWorkflowAdapter's docstring) — cast to
`dict[str, Any]` wherever a test indexes into them more than one level deep,
same reasoning as tests/test_data_quality.py's `cast(dict, ...)`.
"""

from typing import Any, cast

import pytest

from adapters.mock_model_registry_adapter import MockModelRegistryAdapter
from adapters.mock_workflow_adapter import MockWorkflowAdapter


@pytest.fixture
def adapter() -> MockWorkflowAdapter:
    return MockWorkflowAdapter()


def test_trigger_workflow_returns_a_generated_name(adapter: MockWorkflowAdapter) -> None:
    result = cast(
        dict[str, Any],
        adapter.trigger_workflow("train-track-register-golden-path", {"model-name": "x"}),
    )

    name = result["metadata"]["name"]
    assert isinstance(name, str)
    assert name.startswith("train-track-register-golden-path-")


def test_get_workflow_status_transitions_from_running_to_succeeded(
    adapter: MockWorkflowAdapter,
) -> None:
    result = cast(dict[str, Any], adapter.trigger_workflow("train-track-register-golden-path", {}))
    name = result["metadata"]["name"]
    assert isinstance(name, str)

    first = adapter.get_workflow_status(name)
    second = adapter.get_workflow_status(name)

    assert first["phase"] == "Running"
    assert second["phase"] == "Succeeded"


def test_get_workflow_status_for_unknown_workflow(adapter: MockWorkflowAdapter) -> None:
    status = adapter.get_workflow_status("does-not-exist")

    assert status == {
        "name": "does-not-exist",
        "phase": None,
        "message": "workflow not found",
        "started_at": None,
        "finished_at": None,
        "steps": [],
    }


def test_get_workflow_status_sets_finished_at_once_succeeded(
    adapter: MockWorkflowAdapter,
) -> None:
    result = cast(dict[str, Any], adapter.trigger_workflow("train-track-register-golden-path", {}))
    name = result["metadata"]["name"]
    assert isinstance(name, str)

    first = adapter.get_workflow_status(name)
    second = adapter.get_workflow_status(name)

    assert first["finished_at"] is None
    assert second["finished_at"] is not None
    assert second["steps"] == [
        {
            "name": "train",
            "phase": "Succeeded",
            "started_at": second["started_at"],
            "finished_at": second["finished_at"],
        }
    ]


def test_list_workflows_includes_triggered_workflows(adapter: MockWorkflowAdapter) -> None:
    result = cast(dict[str, Any], adapter.trigger_workflow("train-track-register-golden-path", {}))
    name = result["metadata"]["name"]

    summaries = adapter.list_workflows()

    assert any(s["name"] == name and s["phase"] == "Running" for s in summaries)


def test_trigger_workflow_registers_a_model_when_model_registry_is_wired() -> None:
    model_registry = MockModelRegistryAdapter()
    adapter = MockWorkflowAdapter(model_registry=model_registry)

    adapter.trigger_workflow(
        "train-track-register-golden-path",
        {"model-name": "fraud-detection", "task-type": "classification"},
    )

    details = model_registry.get_model_version_details("fraud-detection", "1")
    assert details["tags"]["task_type"] == "classification"


def test_trigger_workflow_without_model_registry_does_not_raise(
    adapter: MockWorkflowAdapter,
) -> None:
    # No model_registry wired (factory.py only wires one when the registry
    # is ALSO mocked) — must no-op, not crash, same as a "Setup Model
    # Monitoring" call whose parameters carry no model-name/task-type.
    adapter.trigger_workflow("monitor-drift-golden-path", {"schedule": "0 * * * *"})


def test_create_cron_workflow_stores_schedule_and_parameters(
    adapter: MockWorkflowAdapter,
) -> None:
    result = cast(
        dict[str, Any],
        adapter.create_cron_workflow(
            "monitor-fraud-detection",
            "0 * * * *",
            "monitor-drift-golden-path",
            {"model-name": "fraud-detection"},
        ),
    )

    cron_workflow = result["cronWorkflow"]
    assert cron_workflow["metadata"]["name"] == "monitor-fraud-detection"
    assert cron_workflow["spec"]["schedule"] == "0 * * * *"
    assert cron_workflow["spec"]["workflowSpec"]["workflowTemplateRef"] == {
        "name": "monitor-drift-golden-path"
    }
