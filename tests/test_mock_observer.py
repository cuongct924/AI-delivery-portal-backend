"""services/orchestration-api/routers/mock_observer.py — calls the route
functions directly, same pattern as the other router tests."""

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from routers.mock_observer import (  # noqa: E402
    DoraDeploymentsQueryRequest,
    DoraQueryRequest,
    DoraSearchScope,
    query_dora_deployments,
    query_dora_metrics,
)

_START = datetime(2026, 8, 1, tzinfo=UTC)
_END = _START + timedelta(days=30)


def _request(**scope: str) -> DoraQueryRequest:
    return DoraQueryRequest(
        searchScope=DoraSearchScope(namespace="default", **scope),
        startTime=_START.isoformat(),
        endTime=_END.isoformat(),
        granularity="daily",
    )


def test_metrics_are_deterministic_for_a_scope() -> None:
    first = query_dora_metrics(_request(project="fraud-detection"))
    second = query_dora_metrics(_request(project="fraud-detection"))

    assert first.summary.deploymentFrequency == second.summary.deploymentFrequency
    assert first.summary.changeFailureRate == second.summary.changeFailureRate


def test_different_scopes_produce_different_metrics() -> None:
    a = query_dora_metrics(_request(project="fraud-detection"))
    b = query_dora_metrics(_request(project="churn-prediction"))

    assert a.summary.deploymentFrequency != b.summary.deploymentFrequency


def test_series_are_zero_filled_for_frequency_and_failure_rate() -> None:
    response = query_dora_metrics(_request())

    assert response.series.deploymentFrequency is not None
    assert response.series.changeFailureRate is not None
    assert len(response.series.deploymentFrequency) == len(response.series.changeFailureRate)
    assert len(response.series.deploymentFrequency) == 30


def test_data_availability_reports_collecting() -> None:
    response = query_dora_metrics(_request())

    assert response.dataAvailability.collecting is True
    assert response.dataAvailability.deliveryEvents is True


def test_metric_filter_drops_unrequested_metrics() -> None:
    request = _request()
    request.metrics = ["deploymentFrequency"]

    response = query_dora_metrics(request)

    assert response.summary.deploymentFrequency is not None
    assert response.summary.leadTime is None
    assert response.summary.changeFailureRate is None
    assert response.summary.mttr is None
    assert response.series.leadTime is None


def test_deployments_use_mlops_template_names_and_respect_limit() -> None:
    request = DoraDeploymentsQueryRequest(
        searchScope=DoraSearchScope(namespace="default"),
        startTime=_START.isoformat(),
        endTime=_END.isoformat(),
        limit=5,
    )

    response = query_dora_deployments(request)

    assert response.totalCount <= 5
    assert all(d.componentName for d in response.deployments)
    assert all(d.environmentName in {"dev", "staging", "prod"} for d in response.deployments)


def test_deployments_sort_order() -> None:
    request = DoraDeploymentsQueryRequest(
        searchScope=DoraSearchScope(namespace="default"),
        startTime=_START.isoformat(),
        endTime=_END.isoformat(),
        sortOrder="asc",
    )

    response = query_dora_deployments(request)
    timestamps = [d.deployedAt for d in response.deployments]

    assert timestamps == sorted(timestamps)


def test_invalid_granularity_is_rejected() -> None:
    with pytest.raises(ValueError):
        DoraQueryRequest(
            searchScope=DoraSearchScope(namespace="default"),
            startTime=_START.isoformat(),
            endTime=_END.isoformat(),
            granularity="hourly",  # type: ignore[arg-type]
        )


def test_real_captured_runs_are_served_when_window_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from routers import mock_observer

    if not mock_observer._load_real_runs():
        pytest.skip("no captured runs file")

    monkeypatch.setattr(mock_observer, "_USE_CAPTURED_RUNS", True)
    request = DoraQueryRequest(
        searchScope=DoraSearchScope(namespace="default", project="telco-fraud-detection"),
        startTime="2026-09-17T00:00:00Z",
        endTime="2026-09-19T00:00:00Z",
        granularity="daily",
    )

    response = query_dora_metrics(request)

    assert response.summary.deploymentFrequency is not None
    assert response.summary.deploymentFrequency.total > 0
    assert response.summary.changeFailureRate is not None
    assert 0 < response.summary.changeFailureRate.rate < 1


def test_mock_source_ignores_captured_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from routers import mock_observer

    monkeypatch.setattr(mock_observer, "_USE_CAPTURED_RUNS", False)
    request = DoraQueryRequest(
        searchScope=DoraSearchScope(namespace="default", project="telco-fraud-detection"),
        startTime="2026-09-17T00:00:00Z",
        endTime="2026-09-19T00:00:00Z",
        granularity="daily",
    )

    response = query_dora_metrics(request)

    # Synthetic generator zero-fills the whole window (2 daily buckets).
    assert response.series.deploymentFrequency is not None
    assert len(response.series.deploymentFrequency) == 2
