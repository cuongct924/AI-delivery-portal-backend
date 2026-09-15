"""Tests adapters/argo_adapter.py — mocks httpx so no real Argo Server is required."""

from unittest.mock import MagicMock, patch

from adapters.argo_adapter import ArgoAdapter


def _mock_response(json_data: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


def test_get_workflow_status_includes_message() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response({"status": {"phase": "Failed", "message": "pod OOMKilled"}})

    with patch("adapters.argo_adapter.httpx.get", return_value=response) as mock_get:
        result = adapter.get_workflow_status("train-abc123")

    mock_get.assert_called_once_with(
        "http://argo.test/api/v1/workflows/default/train-abc123", timeout=10
    )
    assert result == {
        "name": "train-abc123",
        "phase": "Failed",
        "message": "pod OOMKilled",
        "started_at": None,
        "finished_at": None,
        "steps": [],
    }


def test_get_workflow_status_message_defaults_to_none_when_absent() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response({"status": {"phase": "Succeeded"}})

    with patch("adapters.argo_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["message"] is None


def test_get_workflow_status_extracts_started_and_finished_timestamps() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response(
        {
            "status": {
                "phase": "Succeeded",
                "startedAt": "2026-09-15T01:00:00Z",
                "finishedAt": "2026-09-15T01:05:00Z",
            }
        }
    )

    with patch("adapters.argo_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["started_at"] == "2026-09-15T01:00:00Z"
    assert result["finished_at"] == "2026-09-15T01:05:00Z"


def test_get_workflow_status_extracts_pod_node_steps_only() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response(
        {
            "status": {
                "phase": "Succeeded",
                "nodes": {
                    "train-abc123": {
                        "type": "Steps",
                        "displayName": "train-abc123",
                    },
                    "train-abc123-111": {
                        "type": "Pod",
                        "displayName": "train",
                        "phase": "Succeeded",
                        "startedAt": "2026-09-15T01:00:00Z",
                        "finishedAt": "2026-09-15T01:03:00Z",
                    },
                    "train-abc123-222": {
                        "type": "Pod",
                        "displayName": "register",
                        "phase": "Succeeded",
                        "startedAt": "2026-09-15T01:03:00Z",
                        "finishedAt": "2026-09-15T01:05:00Z",
                    },
                },
            }
        }
    )

    with patch("adapters.argo_adapter.httpx.get", return_value=response):
        result = adapter.get_workflow_status("train-abc123")

    assert result["steps"] == [
        {
            "name": "train",
            "phase": "Succeeded",
            "started_at": "2026-09-15T01:00:00Z",
            "finished_at": "2026-09-15T01:03:00Z",
        },
        {
            "name": "register",
            "phase": "Succeeded",
            "started_at": "2026-09-15T01:03:00Z",
            "finished_at": "2026-09-15T01:05:00Z",
        },
    ]


def test_list_workflows_extracts_name_phase_started_at() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response(
        {
            "items": [
                {
                    "metadata": {"name": "train-abc123"},
                    "status": {"phase": "Succeeded", "startedAt": "2026-08-25T00:00:00Z"},
                },
                {
                    "metadata": {"name": "train-def456"},
                    "status": {"phase": "Running", "startedAt": "2026-08-25T00:05:00Z"},
                },
            ]
        }
    )

    with patch("adapters.argo_adapter.httpx.get", return_value=response) as mock_get:
        result = adapter.list_workflows()

    mock_get.assert_called_once_with("http://argo.test/api/v1/workflows/default", timeout=10)
    assert result == [
        {"name": "train-abc123", "phase": "Succeeded", "startedAt": "2026-08-25T00:00:00Z"},
        {"name": "train-def456", "phase": "Running", "startedAt": "2026-08-25T00:05:00Z"},
    ]


def test_list_workflows_returns_empty_list_when_no_items() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response({})

    with patch("adapters.argo_adapter.httpx.get", return_value=response):
        result = adapter.list_workflows()

    assert result == []


def test_create_cron_workflow_posts_the_expected_body() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    response = _mock_response({"metadata": {"name": "monitor-fraud-detection"}})

    with patch("adapters.argo_adapter.httpx.post", return_value=response) as mock_post:
        result = adapter.create_cron_workflow(
            "monitor-fraud-detection",
            "0 * * * *",
            "monitor-drift-golden-path",
            {"model-name": "fraud-detection", "drift-threshold": "0.5"},
        )

    mock_post.assert_called_once()
    url, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
    assert url == "http://argo.test/api/v1/cron-workflows/default"
    cron_workflow = kwargs["json"]["cronWorkflow"]
    assert cron_workflow["metadata"]["name"] == "monitor-fraud-detection"
    assert cron_workflow["spec"]["schedule"] == "0 * * * *"
    assert cron_workflow["spec"]["workflowSpec"]["workflowTemplateRef"] == {
        "name": "monitor-drift-golden-path"
    }
    assert {"name": "model-name", "value": "fraud-detection"} in cron_workflow["spec"][
        "workflowSpec"
    ]["arguments"]["parameters"]
    assert result == {"metadata": {"name": "monitor-fraud-detection"}}


def test_create_cron_workflow_updates_via_put_when_already_exists() -> None:
    adapter = ArgoAdapter(base_url="http://argo.test")
    conflict_response = MagicMock()
    conflict_response.status_code = 409
    updated_response = _mock_response({"metadata": {"name": "monitor-fraud-detection"}})

    with (
        patch("adapters.argo_adapter.httpx.post", return_value=conflict_response),
        patch("adapters.argo_adapter.httpx.put", return_value=updated_response) as mock_put,
    ):
        result = adapter.create_cron_workflow(
            "monitor-fraud-detection", "0 * * * *", "monitor-drift-golden-path", {}
        )

    mock_put.assert_called_once()
    assert (
        mock_put.call_args.args[0]
        == "http://argo.test/api/v1/cron-workflows/default/monitor-fraud-detection"
    )
    assert result == {"metadata": {"name": "monitor-fraud-detection"}}
