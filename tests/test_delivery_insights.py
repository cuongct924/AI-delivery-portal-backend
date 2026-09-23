"""services/orchestration-api/routers/delivery_insights.py — calls the route
functions directly, same pattern as the other router tests. The adapter
singleton is swapped per test via monkeypatch rather than env vars, so tests
don't depend on import order."""

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from routers import delivery_insights  # noqa: E402
from routers.delivery_insights import (  # noqa: E402
    DoraDeploymentsQueryRequest,
    DoraQueryRequest,
    DoraSearchScope,
    query_dora_deployments,
    query_dora_metrics,
)

from adapters.delivery.mock_delivery_observer_adapter import (  # noqa: E402
    MockDeliveryObserverAdapter,
)

_START = datetime(2026, 8, 1, tzinfo=UTC)
_END = _START + timedelta(days=30)


@pytest.fixture(autouse=True)
def _mock_source_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        delivery_insights, "delivery_observer_adapter", MockDeliveryObserverAdapter("mock")
    )


def _request(
    project: str | None = None,
    component: str | None = None,
    environment: str | None = None,
) -> DoraQueryRequest:
    return DoraQueryRequest(
        searchScope=DoraSearchScope(
            namespace="default", project=project, component=component, environment=environment
        ),
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


def test_rework_rate_series_is_zero_filled_like_the_other_metrics() -> None:
    response = query_dora_metrics(_request(project="fraud-detection"))

    assert response.series.reworkRate is not None
    assert response.series.deploymentFrequency is not None
    assert len(response.series.reworkRate) == len(response.series.deploymentFrequency)
    assert all(0 <= point.rate <= 1 for point in response.series.reworkRate)


def test_data_availability_reports_collecting() -> None:
    response = query_dora_metrics(_request())

    assert response.dataAvailability.collecting is True
    assert response.dataAvailability.deliveryEvents is True


def test_change_failure_rate_summary_splits_infra_and_semantic() -> None:
    response = query_dora_metrics(_request(project="fraud-detection"))

    assert response.summary.changeFailureRate is not None
    assert response.summary.changeFailureRate.infraCfr is not None
    assert response.summary.changeFailureRate.semanticCfr is not None


def test_metric_filter_drops_unrequested_metrics() -> None:
    request = _request()
    request.metrics = ["deploymentFrequency"]

    response = query_dora_metrics(request)

    assert response.summary.deploymentFrequency is not None
    assert response.summary.leadTime is None
    assert response.summary.changeFailureRate is None
    assert response.summary.mttr is None
    assert response.series.leadTime is None


def test_deployments_carry_a_workload_type_derived_from_change_type() -> None:
    request = DoraDeploymentsQueryRequest(
        searchScope=DoraSearchScope(namespace="default"),
        startTime=_START.isoformat(),
        endTime=_END.isoformat(),
        limit=40,
    )

    response = query_dora_deployments(request)

    expected = {
        "infra": "service",
        "model": "ml_model",
        "rag_index": "llm_app",
        "prompt": "llm_app",
    }
    assert response.deployments
    for deployment in response.deployments:
        assert deployment.changeType is not None
        assert deployment.workloadType == expected[deployment.changeType]


def test_deployments_filter_by_workload_type() -> None:
    request = DoraDeploymentsQueryRequest(
        searchScope=DoraSearchScope(namespace="default", workloadType="llm_app"),
        startTime=_START.isoformat(),
        endTime=_END.isoformat(),
        limit=40,
    )

    response = query_dora_deployments(request)

    assert response.deployments
    assert all(d.workloadType == "llm_app" for d in response.deployments)


def test_summary_reports_a_rework_rate() -> None:
    response = query_dora_metrics(_request(project="fraud-detection"))

    assert response.summary.reworkRate is not None
    assert 0 <= response.summary.reworkRate.rate <= 1
    assert response.summary.reworkRate.total > 0


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
    assert all(
        d.changeType in {"infra", "model", "rag_index", "prompt"} for d in response.deployments
    )


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
    from adapters.delivery._captured_runs import load_captured_runs

    if not load_captured_runs():
        pytest.skip("no captured runs file")

    monkeypatch.setattr(
        delivery_insights, "delivery_observer_adapter", MockDeliveryObserverAdapter("captured")
    )
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


def test_mock_source_ignores_captured_runs() -> None:
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
