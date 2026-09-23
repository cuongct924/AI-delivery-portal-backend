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

import os
from collections import defaultdict
from typing import Annotated, Final, cast

from auth.thunder import get_current_user
from costs.events import record_cost_event
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


class CostCheckRequest(BaseModel):
    golden_path: str
    stage: str
    artifact: str = ""
    params: dict[str, object] = {}
    # Overrides the configured budget for this check.
    budget_usd: float | None = None
    # "warn" never blocks; "enforce" blocks a fail-level overrun. Defaults to
    # the COST_GATE_MODE env (warn) so enforcement can be flipped centrally.
    mode: str | None = None


class CostCheckResponse(BaseModel):
    allow: bool
    level: str
    estimated_cost: float
    budget: float | None
    reasons: list[str]


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


# Within this multiple of the budget is a warning; beyond it is a fail.
_WARN_RATIO: Final[float] = 1.2
_DEFAULT_BUDGET_USD: Final[float] = 500.0


def _configured_budget() -> float | None:
    raw = os.getenv("COST_BUDGET_USD")
    if raw is None:
        return _DEFAULT_BUDGET_USD
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_BUDGET_USD


@router.post("/check", response_model=CostCheckResponse)
def check_cost(
    request: CostCheckRequest, user: dict = Depends(get_current_user)
) -> CostCheckResponse:
    """Pre-flight cost guardrail: compare a run's estimate against the budget
    and return ok/warn/fail. In `warn` mode it never blocks; in `enforce` mode a
    fail-level overrun sets allow=False so the caller can stop the run."""
    estimated, _ = estimate_golden_path_cost(request.golden_path, request.stage, request.params)
    budget = request.budget_usd if request.budget_usd is not None else _configured_budget()
    mode = request.mode or os.getenv("COST_GATE_MODE", "warn")

    # Record the estimate so the variance view can compare it with the actual
    # cost the run later records (source="estimate" marks it as a projection).
    record_cost_event(
        stage=request.stage,
        artifact_kind="estimate",
        artifact_id=request.artifact or request.golden_path,
        cost_usd=estimated,
        source="estimate",
        run_id=f"{request.golden_path}:{request.artifact}:{request.stage}",
    )

    if budget is None or budget <= 0:
        return CostCheckResponse(
            allow=True,
            level="ok",
            estimated_cost=estimated,
            budget=None,
            reasons=["No budget set"],
        )
    if estimated <= budget:
        return CostCheckResponse(
            allow=True,
            level="ok",
            estimated_cost=estimated,
            budget=budget,
            reasons=[f"Estimate {estimated:.2f} within budget {budget:.2f}"],
        )
    if estimated <= budget * _WARN_RATIO:
        return CostCheckResponse(
            allow=True,
            level="warn",
            estimated_cost=estimated,
            budget=budget,
            reasons=[
                f"Estimate {estimated:.2f} is within "
                f"{int((_WARN_RATIO - 1) * 100)}% over budget {budget:.2f}"
            ],
        )
    return CostCheckResponse(
        allow=mode != "enforce",
        level="fail",
        estimated_cost=estimated,
        budget=budget,
        reasons=[f"Estimate {estimated:.2f} exceeds budget {budget:.2f}"],
    )


class RateSuggestion(BaseModel):
    id: str
    title: str
    detail: str
    saving_pct: float
    stage: str


class RateOptimizationResponse(BaseModel):
    suggestions: list[RateSuggestion]


class CostVarianceRow(BaseModel):
    artifact: str
    estimated: float
    actual: float
    variance_pct: float | None


class CostVarianceResponse(BaseModel):
    rows: list[CostVarianceRow]


@router.get("/variance", response_model=CostVarianceResponse)
def get_cost_variance(
    start_time: Annotated[str, Query(description="ISO 8601 inclusive lower bound")],
    end_time: Annotated[str, Query(description="ISO 8601 inclusive upper bound")],
    user: dict = Depends(get_current_user),
) -> CostVarianceResponse:
    """Estimate-vs-actual variance per artifact: how far the pre-flight estimate
    was from the cost the run actually recorded. Entries whose source contains
    "estimate" are projections; everything else is actual spend."""
    entries = cost_adapter.query_costs(start_time, end_time)
    estimated: dict[str, float] = defaultdict(float)
    actual: dict[str, float] = defaultdict(float)
    for entry in entries:
        if "estimate" in entry["source"]:
            estimated[entry["artifact_id"]] += entry["cost_usd"]
        else:
            actual[entry["artifact_id"]] += entry["cost_usd"]

    rows: list[CostVarianceRow] = []
    for artifact in set(estimated) | set(actual):
        est = estimated.get(artifact, 0.0)
        act = actual.get(artifact, 0.0)
        rows.append(
            CostVarianceRow(
                artifact=artifact,
                estimated=est,
                actual=act,
                variance_pct=((act - est) / est * 100) if est > 0 else None,
            )
        )
    rows.sort(key=lambda r: abs(r.variance_pct or 0), reverse=True)
    return CostVarianceResponse(rows=rows)


@router.get("/rate-optimization", response_model=RateOptimizationResponse)
def get_rate_optimization(
    start_time: Annotated[str, Query(description="ISO 8601 inclusive lower bound")],
    end_time: Annotated[str, Query(description="ISO 8601 inclusive upper bound")],
    user: dict = Depends(get_current_user),
) -> RateOptimizationResponse:
    """Rate-optimization suggestions derived from what the ledger actually
    spent: spot for fault-tolerant training, committed use for steady serving,
    caching/batch for tokens. Complements right-sizing (usage optimization)."""
    entries = cost_adapter.query_costs(start_time, end_time)
    units = {entry.get("unit", "") for entry in entries}
    stages = {entry["stage"] for entry in entries}
    suggestions: list[RateSuggestion] = []
    if "gpu-hour" in units and "build" in stages:
        suggestions.append(
            RateSuggestion(
                id="spot-training",
                title="Run training on spot GPUs",
                detail=(
                    "Training is fault-tolerant — spot/preemptible GPUs cost "
                    "far less than on-demand."
                ),
                saving_pct=65.0,
                stage="build",
            )
        )
    if "gpu-hour" in units and "run" in stages:
        suggestions.append(
            RateSuggestion(
                id="reserved-serving",
                title="Reserve serving capacity",
                detail="Steady serving traffic qualifies for committed-use / reserved pricing.",
                saving_pct=30.0,
                stage="run",
            )
        )
    if "token" in units:
        suggestions.append(
            RateSuggestion(
                id="prompt-caching",
                title="Cache prompts / batch judge calls",
                detail=(
                    "Prompt caching and the batch API cut token spend on "
                    "repeated or non-urgent calls."
                ),
                saving_pct=40.0,
                stage="gate",
            )
        )
    if "cpu-hour" in units and "run" in stages:
        suggestions.append(
            RateSuggestion(
                id="committed-monitoring",
                title="Commit recurring monitoring",
                detail="Recurring monitoring jobs qualify for committed-use pricing.",
                saving_pct=20.0,
                stage="run",
            )
        )
    return RateOptimizationResponse(suggestions=suggestions)
