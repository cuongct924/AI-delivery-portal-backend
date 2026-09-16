"""Tests adapters/openchoreo_workflow_adapter.py — mocks httpx so no real
openchoreo-api/Thunder instance is required. Style mirrors
tests/test_argo_adapter.py."""

from unittest.mock import MagicMock, patch

import pytest

import adapters.openchoreo_workflow_adapter as oc_adapter
from adapters.openchoreo_workflow_adapter import OpenChoreoWorkflowAdapter, _hyphen_to_camel


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.json.return_value = json_data
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def _stub_access_token():
    """Every real HTTP call goes through _headers() -> _get_access_token(),
    which otherwise makes its own httpx.post to Thunder — stub it so tests
    only exercise this adapter's own openchoreo-api calls. Not autouse:
    test_get_access_token_caches_and_refreshes tests the real thing."""
    with patch("adapters.openchoreo_workflow_adapter._get_access_token", return_value="test-token"):
        yield


def test_hyphen_to_camel() -> None:
    assert _hyphen_to_camel("model-name") == "modelName"
    assert _hyphen_to_camel("hidden-size") == "hiddenSize"
    assert _hyphen_to_camel("mode") == "mode"


def test_trigger_workflow_posts_camel_cased_parameters_and_headers(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {"metadata": {"name": "train-register-golden-path-123"}}, status_code=201
    )

    with patch(
        "adapters.openchoreo_workflow_adapter.httpx.post", return_value=response
    ) as mock_post:
        result = adapter.trigger_workflow(
            "train-register-golden-path",
            {"model-name": "fraud-detection", "dataset-uri": "file:///mnt/data/x.csv"},
        )

    mock_post.assert_called_once()
    url = mock_post.call_args.args[0]
    kwargs = mock_post.call_args.kwargs
    assert url == "http://oc.test/api/v1/namespaces/default/workflowruns"
    assert kwargs["headers"] == {"Authorization": "Bearer test-token"}
    body = kwargs["json"]
    assert body["spec"]["workflow"] == {
        "kind": "ClusterWorkflow",
        "name": "train-register-golden-path",
        "parameters": {"modelName": "fraud-detection", "datasetUri": "file:///mnt/data/x.csv"},
    }
    assert body["metadata"]["name"].startswith("train-register-golden-path-")
    assert result == {"metadata": {"name": "train-register-golden-path-123"}}


def test_get_workflow_status_maps_workflow_succeeded_condition(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {
            "status": {
                "conditions": [
                    {"type": "WorkflowSucceeded", "status": "True", "message": "done"},
                ],
                "startedAt": "2026-09-16T00:13:12Z",
                "completedAt": "2026-09-16T00:14:33Z",
            }
        }
    )

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result == {
        "name": "train-abc123",
        "phase": "Succeeded",
        "message": "done",
        "started_at": "2026-09-16T00:13:12Z",
        "finished_at": "2026-09-16T00:14:33Z",
        "steps": [],
    }


def test_get_workflow_status_maps_workflow_failed_condition(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {
            "status": {
                "conditions": [
                    {"type": "WorkflowRunning", "status": "False"},
                    {"type": "WorkflowFailed", "status": "True", "message": "pod OOMKilled"},
                ]
            }
        }
    )

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["phase"] == "Failed"
    assert result["message"] == "pod OOMKilled"


def test_get_workflow_status_maps_workflow_running_condition(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {"status": {"conditions": [{"type": "WorkflowRunning", "status": "True"}]}}
    )

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["phase"] == "Running"


def test_get_workflow_status_defaults_to_pending_with_no_conditions(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response({"status": {}})

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["phase"] == "Pending"
    assert result["message"] is None


def test_get_workflow_status_workload_updated_takes_priority_over_succeeded(
    _stub_access_token,
) -> None:
    """Mirrors openchoreo-workflows-backend's GenericWorkflowService.ts
    deriveWorkflowRunStatus() priority order — WorkloadUpdated is checked
    first (only ever true for component-build workflows, never a
    training/monitoring ClusterWorkflow, but the priority order must match)."""
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {
            "status": {
                "conditions": [
                    {"type": "WorkflowSucceeded", "status": "True"},
                    {"type": "WorkloadUpdated", "status": "True", "message": "workload updated"},
                ]
            }
        }
    )

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("build-abc123")

    assert result["phase"] == "Succeeded"
    assert result["message"] == "workload updated"


def test_get_workflow_status_extracts_step_timings_from_tasks(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {
            "status": {
                "conditions": [{"type": "WorkflowSucceeded", "status": "True"}],
                "tasks": [
                    {
                        "name": "train",
                        "phase": "Succeeded",
                        "startedAt": "2026-09-16T00:13:12Z",
                        "completedAt": "2026-09-16T00:13:49Z",
                    },
                    {
                        "name": "register",
                        "phase": "Succeeded",
                        "startedAt": "2026-09-16T00:13:59Z",
                        "completedAt": "2026-09-16T00:14:07Z",
                    },
                ],
            }
        }
    )

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["steps"] == [
        {
            "name": "train",
            "phase": "Succeeded",
            "started_at": "2026-09-16T00:13:12Z",
            "finished_at": "2026-09-16T00:13:49Z",
        },
        {
            "name": "register",
            "phase": "Succeeded",
            "started_at": "2026-09-16T00:13:59Z",
            "finished_at": "2026-09-16T00:14:07Z",
        },
    ]


def test_list_workflows_derives_phase_per_item(_stub_access_token) -> None:
    adapter = OpenChoreoWorkflowAdapter(base_url="http://oc.test/api/v1")
    response = _mock_response(
        {
            "items": [
                {
                    "metadata": {"name": "train-abc123"},
                    "status": {
                        "conditions": [{"type": "WorkflowSucceeded", "status": "True"}],
                        "startedAt": "2026-08-25T00:00:00Z",
                    },
                },
                {
                    "metadata": {"name": "train-def456"},
                    "status": {
                        "conditions": [{"type": "WorkflowRunning", "status": "True"}],
                        "startedAt": "2026-08-25T00:05:00Z",
                    },
                },
            ]
        }
    )

    with patch("adapters.openchoreo_workflow_adapter.httpx.get", return_value=response) as mock_get:
        result = adapter.list_workflows()

    assert mock_get.call_args.args[0] == "http://oc.test/api/v1/namespaces/default/workflowruns"
    assert result == [
        {"name": "train-abc123", "phase": "Succeeded", "startedAt": "2026-08-25T00:00:00Z"},
        {"name": "train-def456", "phase": "Running", "startedAt": "2026-08-25T00:05:00Z"},
    ]


def test_create_cron_workflow_raises_not_implemented() -> None:
    adapter = OpenChoreoWorkflowAdapter()
    with pytest.raises(NotImplementedError):
        adapter.create_cron_workflow("monitor-fraud-detection", "0 * * * *", "monitor-drift", {})


def test_get_access_token_caches_and_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oc_adapter, "_cached_token", None)
    monkeypatch.setattr(oc_adapter, "_cached_expiry", 0.0)
    token_response = _mock_response({"access_token": "abc123", "expires_in": 3600})

    with patch(
        "adapters.openchoreo_workflow_adapter.httpx.post", return_value=token_response
    ) as mock_post:
        first = oc_adapter._get_access_token()
        second = oc_adapter._get_access_token()

    assert first == "abc123"
    assert second == "abc123"
    mock_post.assert_called_once()
