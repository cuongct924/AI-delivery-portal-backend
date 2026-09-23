"""Cost Insights API — the AI-lifecycle cost ledger behind the Portal's Cost
Insights page.

Unlike the OpenChoreo observer's cost endpoint (which only knows CPU/memory
per namespace/component), this is attributed to the artifact and lifecycle
stage each golden path produced, so the page can answer "what did this model
version cost to build", "what is this prompt's eval spend", and "how much are
we spending on versions that never shipped".

Every golden-path step that spends money POSTs one entry to /costs/ledger;
the page reads /costs/summary. The ledger is append-only — see
adapters/ai_platform/cost_adapter.py.
"""

from collections import defaultdict
from typing import Annotated, Final, cast

from auth.thunder import get_current_user
from costs.pricing import estimate_golden_path_cost
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from adapters.ai_platform.interfaces import CostLedgerEntry
from adapters.factory import get_cost_adapter

router = APIRouter(prefix="/costs", tags=["costs"])

cost_adapter = get_cost_adapter()

_STAGES: Final[tuple[str, ...]] = ("build", "gate", "run")
_DIMENSIONS: Final[tuple[str, ...]] = ("artifact", "team", "domain", "infra")


class RecordCostRequest(BaseModel):
    timestamp: str
    stage: str
    artifact_kind: str
    artifact_id: str
    version: str = ""
    environment: str = ""
    namespace: str = "default"
    team: str = ""
    business_domain: str = ""
    quantity: float = 0.0
    unit: str = ""
    unit_price: float = 0.0
    cost_usd: float
    source: str = ""
    run_id: str = ""


class CostRow(BaseModel):
    key: str
    total: float
    build: float
    gate: float
    run: float


class CostSummaryResponse(BaseModel):
    start_time: str
    end_time: str
    dimension: str
    total_cost: float
    build_cost: float
    gate_cost: float
    run_cost: float
    rows: list[CostRow]


class EstimateCostRequest(BaseModel):
    golden_path: str
    stage: str
    artifact: str = ""
    params: dict[str, object] = {}


class EstimateCostResponse(BaseModel):
    estimated_cost: float
    currency: str = "USD"
    stage: str
    breakdown: dict[str, float]


def _dimension_value(entry: CostLedgerEntry, dimension: str) -> str:
    match dimension:
        case "artifact":
            return entry["artifact_id"]
        case "team":
            return entry["team"] or entry["artifact_id"]
        case "domain":
            return entry["business_domain"] or entry["artifact_id"]
        case _:
            return entry["artifact_id"]


@router.post("/ledger", status_code=201)
def record_cost(
    request: RecordCostRequest, user: dict = Depends(get_current_user)
) -> dict[str, str]:
    """Append one cost event. Called by the golden-path steps that spend money."""
    cost_adapter.record_cost(cast(CostLedgerEntry, request.model_dump()))
    return {"status": "recorded"}


@router.get("/summary", response_model=CostSummaryResponse)
def get_cost_summary(
    start_time: Annotated[str, Query(description="ISO 8601 inclusive lower bound")],
    end_time: Annotated[str, Query(description="ISO 8601 inclusive upper bound")],
    dimension: Annotated[str, Query(description="artifact | team | domain | infra")] = "artifact",
    stage: Annotated[str | None, Query(description="build | gate | run; omit for all")] = None,
    artifact_kind: str | None = None,
    team: str | None = None,
    business_domain: str | None = None,
    environment: str | None = None,
    user: dict = Depends(get_current_user),
) -> CostSummaryResponse:
    """Aggregate the ledger by dimension, with a per-stage breakdown per row.

    The stage filter narrows the whole response; the dimension decides what a
    row is. Both are the same two axes the Cost Insights page exposes.
    """
    if dimension not in _DIMENSIONS:
        dimension = "artifact"

    entries = cost_adapter.query_costs(
        start_time,
        end_time,
        stage=stage,
        artifact_kind=artifact_kind,
        team=team,
        business_domain=business_domain,
        environment=environment,
    )

    totals: dict[str, float] = defaultdict(float)
    per_row: dict[str, dict[str, float]] = defaultdict(
        lambda: {"build": 0.0, "gate": 0.0, "run": 0.0}
    )
    for entry in entries:
        cost = entry["cost_usd"]
        totals["total"] += cost
        if entry["stage"] in _STAGES:
            totals[entry["stage"]] += cost
            per_row[_dimension_value(entry, dimension)][entry["stage"]] += cost

    rows = [
        CostRow(
            key=key,
            total=sum(stages.values()),
            build=stages["build"],
            gate=stages["gate"],
            run=stages["run"],
        )
        for key, stages in per_row.items()
    ]
    rows.sort(key=lambda r: r.total, reverse=True)

    return CostSummaryResponse(
        start_time=start_time,
        end_time=end_time,
        dimension=dimension,
        total_cost=totals["total"],
        build_cost=totals["build"],
        gate_cost=totals["gate"],
        run_cost=totals["run"],
        rows=rows,
    )


@router.post("/estimate", response_model=EstimateCostResponse)
def estimate_cost(
    request: EstimateCostRequest, user: dict = Depends(get_current_user)
) -> EstimateCostResponse:
    """Pre-flight cost estimate for a golden-path run, so the cost is visible in
    the form/output before anything is provisioned (shift-left FinOps). Priced
    from the same rate tables the real ledger uses; the run's own recorded event
    replaces it once it executes."""
    total, breakdown = estimate_golden_path_cost(request.golden_path, request.stage, request.params)
    return EstimateCostResponse(estimated_cost=total, stage=request.stage, breakdown=breakdown)
