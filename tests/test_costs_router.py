"""services/orchestration-api/routers/costs.py — backed by a real
JsonFileCostLedgerAdapter (tests/conftest.py redirects COST_LEDGER_PATH to a
fresh temp file before any router module imports)."""

from typing import cast

import pytest
from routers.costs import (
    CostCheckRequest,
    CostCheckResponse,
    EstimateCostRequest,
    RecordCostRequest,
    check_cost,
    cost_adapter,
    estimate_cost,
    get_cost_summary,
    record_cost,
)

from adapters.ai_platform.cost_adapter import JsonFileCostLedgerAdapter


@pytest.fixture(autouse=True)
def _reset_ledger() -> None:
    """The router's cost_adapter is a process-wide singleton, so entries would
    otherwise leak between tests and skew the aggregate assertions."""
    cast(JsonFileCostLedgerAdapter, cost_adapter)._write([])


def _record(**over) -> None:
    base = {
        "timestamp": "2026-07-01T00:00:00.000Z",
        "stage": "build",
        "artifact_kind": "model",
        "artifact_id": "fraud-detection",
        "version": "1",
        "environment": "dev",
        "team": "mlops-team",
        "business_domain": "fraud-risk",
        "quantity": 2.0,
        "unit": "gpu-hour",
        "unit_price": 1.5,
        "cost_usd": 3.0,
        "source": "mlflow",
        "run_id": "run-1",
    }
    record_cost(RecordCostRequest(**{**base, **over}))


def test_record_cost_appends_to_the_ledger() -> None:
    _record(run_id="record-only")
    summary = get_cost_summary(
        start_time="2026-07-01T00:00:00.000Z",
        end_time="2026-07-02T00:00:00.000Z",
        dimension="artifact",
    )
    assert summary.total_cost >= 3.0


def test_summary_breaks_down_by_stage() -> None:
    _record(stage="build", cost_usd=10.0, run_id="b")
    _record(stage="gate", cost_usd=2.0, run_id="g")
    _record(stage="run", cost_usd=5.0, run_id="r")

    summary = get_cost_summary(
        start_time="2026-07-01T00:00:00.000Z",
        end_time="2026-07-02T00:00:00.000Z",
        dimension="artifact",
    )
    assert summary.build_cost == 10.0
    assert summary.gate_cost == 2.0
    assert summary.run_cost == 5.0
    assert summary.total_cost == 17.0


def test_summary_groups_rows_by_dimension() -> None:
    _record(artifact_id="model-a", team="team-a", cost_usd=4.0, run_id="a")
    _record(artifact_id="model-b", team="team-b", cost_usd=6.0, run_id="b")

    by_team = get_cost_summary(
        start_time="2026-07-01T00:00:00.000Z",
        end_time="2026-07-02T00:00:00.000Z",
        dimension="team",
    )
    assert {row.key for row in by_team.rows} == {"team-a", "team-b"}
    # Highest spend first.
    assert by_team.rows[0].key == "team-b"


def test_summary_stage_filter_narrows_the_whole_response() -> None:
    _record(stage="build", cost_usd=10.0, run_id="b")
    _record(stage="run", cost_usd=5.0, run_id="r")

    summary = get_cost_summary(
        start_time="2026-07-01T00:00:00.000Z",
        end_time="2026-07-02T00:00:00.000Z",
        dimension="artifact",
        stage="run",
    )
    assert summary.total_cost == 5.0
    assert summary.build_cost == 0.0


def test_summary_unknown_dimension_falls_back_to_artifact() -> None:
    _record(artifact_id="fallback-model", run_id="f")
    summary = get_cost_summary(
        start_time="2026-07-01T00:00:00.000Z",
        end_time="2026-07-02T00:00:00.000Z",
        dimension="nonsense",
    )
    assert summary.dimension == "artifact"
    assert any(row.key == "fallback-model" for row in summary.rows)


def test_estimate_cost_prices_gpu_serving() -> None:
    response = estimate_cost(
        EstimateCostRequest(
            golden_path="llm-serve-deploy",
            stage="run",
            params={"gpuType": "H100", "gpuCount": 2},
        )
    )
    assert response.estimated_cost == 2 * 24 * 4.50
    assert response.breakdown["gpu"] == 2 * 24 * 4.50
    assert response.stage == "run"


def test_estimate_cost_scales_training_with_epochs() -> None:
    small = estimate_cost(
        EstimateCostRequest(golden_path="train-track-register", stage="build", params={"epochs": 1})
    )
    big = estimate_cost(
        EstimateCostRequest(
            golden_path="train-track-register", stage="build", params={"epochs": 10}
        )
    )
    assert big.estimated_cost > small.estimated_cost


def test_estimate_cost_unknown_path_falls_back() -> None:
    response = estimate_cost(EstimateCostRequest(golden_path="nope", stage="run"))
    assert response.estimated_cost > 0


def _check(**over) -> CostCheckResponse:
    base = {
        "golden_path": "llm-serve-deploy",
        "stage": "run",
        "params": {"gpuType": "H100", "gpuCount": 1},
    }
    return check_cost(CostCheckRequest(**{**base, **over}))


def test_check_cost_ok_within_budget() -> None:
    response = _check(budget_usd=1000)
    assert response.level == "ok"
    assert response.allow is True


def test_check_cost_warns_just_over_budget() -> None:
    # H100 x1 x24h = 108; budget 100 is within the 20% warn band.
    response = _check(budget_usd=100)
    assert response.level == "warn"
    assert response.allow is True


def test_check_cost_fails_far_over_budget() -> None:
    response = _check(budget_usd=50)
    assert response.level == "fail"
    # warn mode still allows.
    assert response.allow is True


def test_check_cost_enforce_blocks_a_fail() -> None:
    response = _check(budget_usd=50, mode="enforce")
    assert response.level == "fail"
    assert response.allow is False
