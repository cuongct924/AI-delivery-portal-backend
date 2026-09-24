"""adapters/delivery/argo_cron_workflow_adapter.py — patches the k8s client
and kubeconfig bootstrap, never touches a real cluster."""

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException

from adapters.delivery.argo_cron_workflow_adapter import ArgoCronWorkflowAdapter


def _adapter(mock_api: MagicMock) -> ArgoCronWorkflowAdapter:
    with (
        patch("adapters.delivery.argo_cron_workflow_adapter.load_kube_config_once"),
        patch(
            "adapters.delivery.argo_cron_workflow_adapter.client.CustomObjectsApi",
            return_value=mock_api,
        ),
    ):
        return ArgoCronWorkflowAdapter()


def test_create_cron_workflow_posts_an_inline_monitoring_spec() -> None:
    mock_api = MagicMock()
    mock_api.create_namespaced_custom_object.return_value = {"metadata": {"name": "monitor-x"}}
    adapter = _adapter(mock_api)

    result = adapter.create_cron_workflow(
        "monitor-x",
        "0 0 * * *",
        "monitor-drift-golden-path",
        {
            "model-name": "x",
            "model-version": "2",
            "reference-data-uri": "file:///mnt/data/x/train.csv",
            "production-data-uri": "file:///mnt/data/x/prod.csv",
            "monitoring-type": "data-drift",
            "drift-threshold": "0.5",
            "metric-names": "f1_score,recall",
            "min-metric-threshold": "0.85",
            "on-drift-detected": "alert-only",
        },
    )

    _, _, namespace, _, body = mock_api.create_namespaced_custom_object.call_args.args
    assert namespace == "workflows-default"
    assert body["spec"]["schedules"] == ["0 0 * * *"]
    assert result == {"cronWorkflow": {"metadata": {"name": "monitor-x"}}}

    spec = body["spec"]["workflowSpec"]
    assert spec["serviceAccountName"] == "workflow-sa"
    assert spec["entrypoint"] == "monitor"
    template = spec["templates"][0]
    assert template["container"]["image"] == "training-image:local"
    assert template["container"]["command"] == ["python", "monitor_drift.py"]
    assert template["nodeSelector"] == {"plane.viettel.vn": "ai-platform-workflow"}
    env = {e["name"]: e["value"] for e in template["container"]["env"]}
    assert env["MODEL_NAME"] == "x"
    assert env["MODEL_VERSION"] == "2"
    assert env["REFERENCE_DATA_URI"] == "file:///mnt/data/x/train.csv"
    assert env["PRODUCTION_DATA_URI"] == "file:///mnt/data/x/prod.csv"
    assert env["METRIC_NAMES"] == "f1_score,recall"
    # Pods always run in-cluster — never the dev host's localhost URLs.
    assert env["MLFLOW_TRACKING_URI"].startswith("http://mlflow.ai-platform-zone")
    assert "orchestration-api" in env["ORCHESTRATION_API_URL"]
    # Optionals absent from parameters stay absent from env.
    assert "GROUND_TRUTH_DATA_URI" not in env
    assert "RETRAIN_REQUEST_JSON" not in env


def test_create_cron_workflow_replaces_spec_when_schedule_already_exists() -> None:
    mock_api = MagicMock()
    mock_api.create_namespaced_custom_object.side_effect = ApiException(status=409)
    mock_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "monitor-x", "resourceVersion": "123"},
        "spec": {"schedule": "0 * * * *"},
    }
    mock_api.replace_namespaced_custom_object.return_value = {"metadata": {"name": "monitor-x"}}
    adapter = _adapter(mock_api)

    adapter.create_cron_workflow("monitor-x", "0 0 * * *", "monitor-drift-golden-path", {})

    _, _, _, _, name, current = mock_api.replace_namespaced_custom_object.call_args.args
    assert name == "monitor-x"
    assert current["spec"]["schedules"] == ["0 0 * * *"]
    assert current["metadata"]["resourceVersion"] == "123"


def test_create_cron_workflow_reraises_non_conflict_errors() -> None:
    mock_api = MagicMock()
    mock_api.create_namespaced_custom_object.side_effect = ApiException(status=403)
    adapter = _adapter(mock_api)

    with pytest.raises(ApiException):
        adapter.create_cron_workflow("monitor-x", "0 0 * * *", "monitor-drift-golden-path", {})
