"""services/orchestration-api/routers/monitoring.py — same pattern as
tests/test_recommendations_router.py: patches the module-level
`workflow_adapter` singleton and calls route functions directly."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from routers.monitoring import SetupMonitoringRequest, setup_monitoring  # noqa: E402


def test_setup_monitoring_creates_a_deterministically_named_cron_workflow() -> None:
    request = SetupMonitoringRequest(
        model_name="fraud-detection",
        model_version="3",
        reference_data_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
        production_data_uri="file:///mnt/monitoring/fraud-detection-recent.csv",
        schedule="0 * * * *",
    )
    with patch("routers.monitoring.workflow_adapter") as mock_argo:
        response = setup_monitoring(request)

    mock_argo.create_cron_workflow.assert_called_once_with(
        "monitor-fraud-detection",
        "0 * * * *",
        "monitor-drift-golden-path",
        {
            "model-name": "fraud-detection",
            "model-version": "3",
            "reference-data-uri": "file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
            "production-data-uri": "file:///mnt/monitoring/fraud-detection-recent.csv",
            "drift-threshold": "0.5",
            "on-drift-detected": "alert-only",
        },
    )
    assert response.cron_workflow_name == "monitor-fraud-detection"


def test_setup_monitoring_forwards_retrain_request_json_for_auto_retrain() -> None:
    request = SetupMonitoringRequest(
        model_name="fraud-detection",
        model_version="3",
        reference_data_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
        production_data_uri="file:///mnt/monitoring/fraud-detection-recent.csv",
        schedule="0 0 * * *",
        on_drift_detected="auto-retrain",
        retrain_request_json='{"model_name": "fraud-detection"}',
    )
    with patch("routers.monitoring.workflow_adapter") as mock_argo:
        setup_monitoring(request)

    call_args = mock_argo.create_cron_workflow.call_args.args
    assert call_args[3]["retrain-request-json"] == '{"model_name": "fraud-detection"}'


def test_setup_monitoring_requires_retrain_request_json_for_auto_retrain() -> None:
    request = SetupMonitoringRequest(
        model_name="fraud-detection",
        model_version="3",
        reference_data_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
        production_data_uri="file:///mnt/monitoring/fraud-detection-recent.csv",
        schedule="0 0 * * *",
        on_drift_detected="auto-retrain",
    )
    with pytest.raises(ValueError, match="retrain_request_json is required"):
        setup_monitoring(request)
