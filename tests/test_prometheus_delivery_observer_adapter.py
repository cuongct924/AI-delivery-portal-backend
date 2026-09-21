"""Tests adapters/delivery/prometheus_delivery_observer_adapter.py.

httpx is mocked — no live cluster/Prometheus needed. The model registry
adapter is a MagicMock returning an empty DataFrame by default (no captured
runs to enrich in these tests' fixed window, matched to no MLflow drift
runs either), exercising the "no data available" paths honestly rather than
fabricating values.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from adapters.delivery.interfaces import DeliveryScope
from adapters.delivery.prometheus_delivery_observer_adapter import PrometheusDeliveryObserverAdapter

_START = datetime(2026, 8, 1, tzinfo=UTC)
_END = datetime(2026, 8, 3, tzinfo=UTC)  # 2 daily buckets


def _scope() -> DeliveryScope:
    return DeliveryScope(namespace="default", project=None, component=None, environment=None)


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


def test_query_metrics_builds_series_from_prometheus_response(
    model_registry_adapter: MagicMock,
) -> None:
    adapter = PrometheusDeliveryObserverAdapter(
        model_registry_adapter=model_registry_adapter, prometheus_url="http://prom.test:9090"
    )
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


def test_query_metrics_handles_no_prometheus_data(model_registry_adapter: MagicMock) -> None:
    adapter = PrometheusDeliveryObserverAdapter(model_registry_adapter=model_registry_adapter)
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


def test_query_metrics_applies_metric_filter(model_registry_adapter: MagicMock) -> None:
    adapter = PrometheusDeliveryObserverAdapter(model_registry_adapter=model_registry_adapter)
    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.httpx.get",
        return_value=_matrix_response([1.0, 1.0]),
    ):
        result = adapter.query_metrics(_scope(), _START, _END, "daily", ["deploymentFrequency"])

    assert result["summary"]["deploymentFrequency"] is not None
    assert result["summary"]["leadTime"] is None
    assert result["summary"]["changeFailureRate"] is None
    assert result["summary"]["mttr"] is None


def test_query_deployments_returns_empty_when_no_captured_runs(
    model_registry_adapter: MagicMock,
) -> None:
    adapter = PrometheusDeliveryObserverAdapter(model_registry_adapter=model_registry_adapter)

    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.load_captured_runs",
        return_value=[],
    ):
        result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    assert result["deployments"] == []
    assert result["totalCount"] == 0
    model_registry_adapter.search_runs.assert_not_called()


def test_query_deployments_enriches_rows_with_drift_score(
    model_registry_adapter: MagicMock,
) -> None:
    adapter = PrometheusDeliveryObserverAdapter(model_registry_adapter=model_registry_adapter)
    captured_runs = [
        {
            "name": "train-track-register-abc12",
            "started_at": "2026-08-01T10:00:00Z",
            "finished_at": "2026-08-01T11:00:00Z",
            "outcome": "success",
            "steps": [],
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

    with patch(
        "adapters.delivery.prometheus_delivery_observer_adapter.load_captured_runs",
        return_value=captured_runs,
    ):
        result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    assert len(result["deployments"]) == 1
    row = result["deployments"][0]
    assert row["driftScore"] == pytest.approx(0.42)
    assert row["driftTriggered"] is True
    assert row["recoveryStrategy"] == "retrain"
