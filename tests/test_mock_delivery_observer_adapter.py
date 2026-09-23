"""Tests adapters/delivery/mock_delivery_observer_adapter.py."""

from datetime import UTC, datetime, timedelta

from adapters.delivery.interfaces import DeliveryScope, WorkloadType
from adapters.delivery.mock_delivery_observer_adapter import MockDeliveryObserverAdapter

_START = datetime(2026, 8, 1, tzinfo=UTC)
_END = _START + timedelta(days=30)


def _scope(project: str | None = None, workload_type: WorkloadType | None = None) -> DeliveryScope:
    return DeliveryScope(
        namespace="default",
        project=project,
        component=None,
        environment=None,
        workloadType=workload_type,
    )


def test_query_metrics_is_deterministic_per_seed() -> None:
    adapter = MockDeliveryObserverAdapter("mock")
    scope = _scope(project="fraud-detection")

    first = adapter.query_metrics(scope, _START, _END, "daily", None)
    second = adapter.query_metrics(scope, _START, _END, "daily", None)

    assert first["summary"]["deploymentFrequency"] == second["summary"]["deploymentFrequency"]


def test_query_metrics_cfr_summary_has_infra_semantic_split() -> None:
    adapter = MockDeliveryObserverAdapter("mock")
    result = adapter.query_metrics(_scope(project="fraud-detection"), _START, _END, "daily", None)

    cfr = result["summary"]["changeFailureRate"]
    assert cfr is not None
    assert cfr["infraCfr"] is not None
    assert cfr["semanticCfr"] is not None


def test_query_deployments_populates_ml_fields_for_non_infra_rows() -> None:
    adapter = MockDeliveryObserverAdapter("mock")
    result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    ml_rows = [d for d in result["deployments"] if d["changeType"] != "infra"]
    breakdowns = [d["leadTimeBreakdown"] for d in ml_rows]
    assert ml_rows
    assert all(d["evalScore"] is not None for d in ml_rows)
    assert all(breakdown is not None for breakdown in breakdowns)
    assert all(
        set(breakdown) == {"data_prep", "train", "eval", "deploy"}
        for breakdown in breakdowns
        if breakdown is not None
    )


def test_query_deployments_version_field_matches_change_type() -> None:
    adapter = MockDeliveryObserverAdapter("mock")
    result = adapter.query_deployments(_scope(), _START, _END, 100, "desc")

    for row in result["deployments"]:
        if row["changeType"] == "model":
            assert row["modelVersion"] is not None
            assert row["promptVersion"] is None
            assert row["ragIndexVersion"] is None
        elif row["changeType"] == "prompt":
            assert row["promptVersion"] is not None
            assert row["modelVersion"] is None
        elif row["changeType"] == "rag_index":
            assert row["ragIndexVersion"] is not None
            assert row["modelVersion"] is None
        else:
            assert row["modelVersion"] is None
            assert row["promptVersion"] is None
            assert row["ragIndexVersion"] is None


def test_query_metrics_metric_filter_drops_unrequested() -> None:
    adapter = MockDeliveryObserverAdapter("mock")
    result = adapter.query_metrics(_scope(), _START, _END, "daily", ["deploymentFrequency"])

    assert result["summary"]["deploymentFrequency"] is not None
    assert result["summary"]["leadTime"] is None
    assert result["summary"]["changeFailureRate"] is None
    assert result["summary"]["mttr"] is None
