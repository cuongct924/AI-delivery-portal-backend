"""adapters/ai_platform/cost_adapter.py — exercises JsonFileCostLedgerAdapter
against a real file on tmp_path (no SDK to mock, same convention as
test_version_registry_adapter.py)."""

from adapters.ai_platform.cost_adapter import JsonFileCostLedgerAdapter
from adapters.ai_platform.interfaces import CostLedgerEntry


def _adapter(tmp_path) -> JsonFileCostLedgerAdapter:
    return JsonFileCostLedgerAdapter(path=str(tmp_path / "cost-ledger.json"))


def _entry(over: dict | None = None) -> CostLedgerEntry:
    base: CostLedgerEntry = {
        "timestamp": "2026-07-01T00:00:00.000Z",
        "stage": "build",
        "artifact_kind": "model",
        "artifact_id": "fraud-detection",
        "version": "1",
        "environment": "dev",
        "namespace": "default",
        "team": "mlops-team",
        "business_domain": "fraud-risk",
        "quantity": 2.0,
        "unit": "gpu-hour",
        "unit_price": 1.5,
        "cost_usd": 3.0,
        "source": "mlflow",
        "run_id": "run-1",
    }
    return {**base, **(over or {})}  # type: ignore[return-value]


def test_record_then_query_round_trips(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.record_cost(_entry())

    results = adapter.query_costs("2026-07-01T00:00:00.000Z", "2026-07-02T00:00:00.000Z")
    assert len(results) == 1
    assert results[0]["artifact_id"] == "fraud-detection"


def test_query_is_append_only_and_keeps_every_entry(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.record_cost(_entry({"run_id": "run-1"}))
    adapter.record_cost(_entry({"run_id": "run-2"}))

    results = adapter.query_costs("2026-07-01T00:00:00.000Z", "2026-07-02T00:00:00.000Z")
    assert {r["run_id"] for r in results} == {"run-1", "run-2"}


def test_query_filters_by_window(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.record_cost(_entry({"timestamp": "2026-06-01T00:00:00.000Z"}))
    adapter.record_cost(_entry({"timestamp": "2026-07-01T00:00:00.000Z"}))

    results = adapter.query_costs("2026-07-01T00:00:00.000Z", "2026-07-02T00:00:00.000Z")
    assert len(results) == 1


def test_query_filters_by_stage_and_dimensions(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.record_cost(_entry({"stage": "build", "team": "mlops-team"}))
    adapter.record_cost(_entry({"stage": "run", "team": "llmops-team"}))

    assert (
        len(
            adapter.query_costs("2026-07-01T00:00:00.000Z", "2026-07-02T00:00:00.000Z", stage="run")
        )
        == 1
    )
    assert (
        len(
            adapter.query_costs(
                "2026-07-01T00:00:00.000Z",
                "2026-07-02T00:00:00.000Z",
                team="mlops-team",
            )
        )
        == 1
    )


def test_query_on_missing_file_returns_empty(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.query_costs("2026-07-01T00:00:00.000Z", "2026-07-02T00:00:00.000Z") == []
