"""Tests adapters/delivery/prometheus_delivery_observer_adapter.py.

httpx is mocked — no live cluster/Prometheus needed. The model registry
adapter and deployment event store are MagicMocks; query_deployments tests
exercise SqliteDeploymentEventStore.list_events's contract via mocking
rather than a real sqlite file (that round-trip lives in
tests/test_deployment_event_store.py).
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from adapters.delivery.interfaces import DeliveryScope, WorkloadType
from adapters.delivery.prometheus_delivery_observer_adapter import PrometheusDeliveryObserverAdapter

_START = datetime(2026, 8, 1, tzinfo=UTC)
_END = datetime(2026, 8, 3, tzinfo=UTC)  # 2 daily buckets


def _scope(workload_type: WorkloadType | None = None) -> DeliveryScope:
    return DeliveryScope(
        namespace="default",
        project=None,
        component=None,
        environment=None,
        workloadType=workload_type,
    )


def _matrix_response(values: list[float]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": {
            "result": [
                {
                    "metric": {},
                    "values": [[1700000000 + i * 86400, str(v)] for i, v in enumerate(values)],
                }
            ]
        }
    }
    return response


@pytest.fixture
def model_registry_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.search_runs.return_value = pd.DataFrame()
    return adapter


@pytest.fixture
def deployment_event_store() -> MagicMock:
    store = MagicMock()
    store.list_events.return_value = []
    return store


def _adapter(
    model_registry_adapter: MagicMock,
    deployment_event_store: MagicMock,
    prometheus_url: str | None = None,
) -> PrometheusDeliveryObserverAdapter:
    return PrometheusDeliveryObserverAdapter(
        model_registry_adapter=model_registry_adapter,
        deployment_event_store=deployment_event_store,
        prometheus_url=prometheus_url,
    )


def test_query_metrics_builds_series_from_prometheus_response(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    adapter = _adapter(model_registry_adapter, deployment_event_store, "http://prom.test:9090")
    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.httpx.get",
        return_value=_matrix_response([3.0, 5.0]),
    ) as mock_get:
        result = adapter.query_metrics(_scope(), _START, _END, "daily", None)

    assert mock_get.called
    freq_series = result["series"]["deploymentFrequency"]
    freq_summary = result["summary"]["deploymentFrequency"]
    assert freq_series is not None
    assert [p["count"] for p in freq_series] == [3, 5]
    assert freq_summary is not None
    assert freq_summary["total"] == 8


def test_query_metrics_handles_no_prometheus_data(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    adapter = _adapter(model_registry_adapter, deployment_event_store)
    empty_response = MagicMock()
    empty_response.raise_for_status.return_value = None
    empty_response.json.return_value = {"data": {"result": []}}

    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.httpx.get",
        return_value=empty_response,
    ):
        result = adapter.query_metrics(_scope(), _START, _END, "daily", None)

    freq_summary = result["summary"]["deploymentFrequency"]
    assert freq_summary is not None
    assert freq_summary["total"] == 0
    assert result["dataAvailability"]["deliveryEvents"] is False


def test_query_metrics_computes_rework_rate_from_rollback_events(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    adapter = _adapter(model_registry_adapter, deployment_event_store)
    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.httpx.get",
        return_value=_matrix_response([3.0, 5.0]),
    ):
        result = adapter.query_metrics(_scope(), _START, _END, "daily", None)

    rework = result["summary"]["reworkRate"]
    assert rework is not None
    assert rework["reworked"] == 8
    assert rework["total"] == 8
    assert rework["rate"] == pytest.approx(1.0)

    series = result["series"]["reworkRate"]
    assert series is not None
    assert [p["reworked"] for p in series] == [3, 5]
    assert [p["total"] for p in series] == [3, 5]
    assert all(p["rate"] == pytest.approx(1.0) for p in series)


def test_query_metrics_applies_metric_filter(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    adapter = _adapter(model_registry_adapter, deployment_event_store)
    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.httpx.get",
        return_value=_matrix_response([1.0, 1.0]),
    ):
        result = adapter.query_metrics(_scope(), _START, _END, "daily", ["deploymentFrequency"])

    assert result["summary"]["deploymentFrequency"] is not None
    assert result["summary"]["leadTime"] is None
    assert result["summary"]["changeFailureRate"] is None
    assert result["summary"]["mttr"] is None
    assert result["summary"]["reworkRate"] is None


def test_query_deployments_returns_empty_when_store_has_no_events(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    adapter = _adapter(model_registry_adapter, deployment_event_store)

    result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    assert result["deployments"] == []
    assert result["totalCount"] == 0
    model_registry_adapter.search_runs.assert_not_called()


def test_query_deployments_enriches_model_rows_with_drift_score(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    deployment_event_store.list_events.return_value = [
        {
            "name": "train-track-register-abc12",
            "changeType": "model",
            "projectName": "fraud-detection",
            "componentName": "train-track-register",
            "environmentName": "development",
            "outcome": "success",
            "startedAt": "2026-08-01T10:00:00",
            "finishedAt": "2026-08-01T11:00:00",
            "steps": None,
        }
    ]
    model_registry_adapter.search_runs.return_value = pd.DataFrame(
        [
            {
                "start_time": pd.Timestamp("2026-08-01T09:00:00Z"),
                "metrics.drift_share": 0.42,
                "tags.drift_detected": "True",
                "params.on_drift_detected": "auto-retrain",
            }
        ]
    )
    adapter = _adapter(model_registry_adapter, deployment_event_store)

    result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    assert len(result["deployments"]) == 1
    row = result["deployments"][0]
    assert row["driftScore"] == pytest.approx(0.42)
    assert row["driftTriggered"] is True
    assert row["recoveryStrategy"] == "retrain"


def test_query_deployments_skips_drift_lookup_for_non_model_rows(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    deployment_event_store.list_events.return_value = [
        {
            "name": "rag-activate-collection-1",
            "changeType": "rag_index",
            "projectName": "collection",
            "componentName": "rag-index",
            "environmentName": "development",
            "outcome": "success",
            "startedAt": "2026-08-01T10:00:00",
            "finishedAt": "2026-08-01T10:00:00",
            "steps": None,
        }
    ]
    adapter = _adapter(model_registry_adapter, deployment_event_store)

    result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    assert result["deployments"][0]["changeType"] == "rag_index"
    model_registry_adapter.search_runs.assert_not_called()


def test_query_deployments_derives_workload_type_from_change_type(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    deployment_event_store.list_events.return_value = [
        {
            "name": "prompt-activate-1",
            "changeType": "prompt",
            "projectName": "support-copilot",
            "componentName": "prompt",
            "environmentName": "development",
            "outcome": "success",
            "startedAt": "2026-08-01T10:00:00",
            "finishedAt": "2026-08-01T10:00:00",
            "steps": None,
        }
    ]
    adapter = _adapter(model_registry_adapter, deployment_event_store)

    result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    assert result["deployments"][0]["workloadType"] == "llm_app"


def test_query_deployments_filters_by_workload_type(
    model_registry_adapter: MagicMock, deployment_event_store: MagicMock
) -> None:
    deployment_event_store.list_events.return_value = [
        {
            "name": "prompt-activate-1",
            "changeType": "prompt",
            "projectName": "support-copilot",
            "componentName": "prompt",
            "environmentName": "development",
            "outcome": "success",
            "startedAt": "2026-08-01T10:00:00",
            "finishedAt": "2026-08-01T10:00:00",
            "steps": None,
        },
        {
            "name": "train-track-register-abc12",
            "changeType": "model",
            "projectName": "fraud-detection",
            "componentName": "train-track-register",
            "environmentName": "development",
            "outcome": "success",
            "startedAt": "2026-08-01T10:00:00",
            "finishedAt": "2026-08-01T11:00:00",
            "steps": None,
        },
    ]
    adapter = _adapter(model_registry_adapter, deployment_event_store)

    result = adapter.query_deployments(_scope(workload_type="ml_model"), _START, _END, 100, "desc")

    assert len(result["deployments"]) == 1
    assert result["deployments"][0]["changeType"] == "model"
