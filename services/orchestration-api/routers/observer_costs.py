"""Mock OpenChoreo observer cost API for the Cost Insights page.

The Portal's Cost Insights page calls the OpenChoreo observer directly
(`/api/v1alpha1/costs/namespaces/{ns}/environments/{env}` and
`.../recommendations`). Without a real observability plane those calls 404, so
this router reproduces the contract with deterministic synthetic data — the
same role routers/delivery_insights.py plays for the DORA API.

Point the frontend's observer URL at orchestration-api (frontend config
`openchoreo.observability.mockObserverUrl`) and the page renders end to end.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Annotated, Final, cast

from fastapi import APIRouter, Query
from pydantic import BaseModel

from adapters.ai_platform.opencost_adapter import fetch_allocation, to_cost_items
from adapters.factory import get_cost_adapter

router = APIRouter(prefix="/api/v1alpha1/costs", tags=["observer-costs"])

cost_adapter = get_cost_adapter()

# Synthetic scope used when the caller doesn't narrow to a project/component —
# enough rows for the table and charts to be meaningful.
_DEMO_PROJECTS: Final[tuple[str, ...]] = (
    "ai-delivery-portal",
    "telco-fraud-detection",
    "customer-segmentation",
)
_DEMO_COMPONENTS: Final[tuple[str, ...]] = (
    "serving",
    "training",
    "monitoring",
    "rag-index",
)

# Demo attribution so the persona views (artifact/team/domain) group by
# distinct values instead of all collapsing onto the project.
_DEMO_ATTRIBUTION: Final[dict[str, tuple[str, str]]] = {
    "ai-delivery-portal": ("platform-team", "network-infrastructure"),
    "telco-fraud-detection": ("fraud-risk-team", "fraud-risk"),
    "customer-segmentation": ("marketing-team", "marketing-sales"),
}

_GRANULARITY_HOURS: Final[dict[str, int]] = {
    "1h": 1,
    "6h": 6,
    "12h": 12,
    "1d": 24,
    "7d": 24 * 7,
}


class CostItem(BaseModel):
    component: str
    startTime: str
    endTime: str
    environment: str
    project: str
    namespace: str
    cpuCost: float
    memoryCost: float
    efficiency: float
    # AI-lifecycle attribution, present on ledger-derived items only. The
    # frontend's stage/dimension switchers read these; absent means "run".
    stage: str | None = None
    team: str | None = None
    businessDomain: str | None = None
    artifact: str | None = None
    # Cost split by resource, so the frontend's AI cost model (cpu+memory+gpu+
    # token) is populated instead of everything landing in cpuCost. Present on
    # ledger-derived items only.
    gpuCost: float | None = None
    tokenCost: float | None = None
    # Usage counters driving unit economics (cost per 1k tokens).
    usage: dict[str, float] | None = None
    # GPU utilization ratio (0..1) for GPU-backed components, for the
    # resource-utilization-efficiency KPI. Absent for CPU-only items.
    gpuUtilization: float | None = None


class CostResourceProfile(BaseModel):
    cpuRequest: str
    cpuLimit: str
    memoryRequest: str
    memoryLimit: str
    cpuCost: float
    memoryCost: float


class CostRecommendationItem(BaseModel):
    component: str
    environment: str
    project: str
    namespace: str
    current: CostResourceProfile
    recommendation: CostResourceProfile


class CostItemsResponse(BaseModel):
    items: list[CostItem]


class CostRecommendationsResponse(BaseModel):
    items: list[CostRecommendationItem]


def _seed(*parts: str) -> int:
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _unit(seed: int, lo: float, hi: float) -> float:
    return lo + (seed % 1000) / 1000 * (hi - lo)


def _parse(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return fallback


def _targets(project: str | None, component: str | None) -> list[tuple[str, str]]:
    """(project, component) pairs the response should cover for this scope."""
    if component:
        return [(project or _DEMO_PROJECTS[0], component)]
    if project:
        return [(project, c) for c in _DEMO_COMPONENTS]
    return [(p, c) for p in _DEMO_PROJECTS for c in _DEMO_COMPONENTS[:2]]


def _item(
    namespace: str,
    environment: str,
    project: str,
    component: str,
    start: datetime,
    end: datetime,
    scale: float = 1.0,
) -> CostItem:
    seed = _seed(namespace, environment, project, component)
    team, domain = _DEMO_ATTRIBUTION.get(project, ("platform-team", "network-infrastructure"))
    # Serving components carry an inference count so the unit-economics metric
    # (cost per 1k inferences) has a denominator.
    usage = {"inferences": float(1000 + seed % 9000)} if component == "serving" else None
    # GPU-backed components carry a utilization ratio for the efficiency KPI.
    gpu_utilization = (
        round(_unit(seed + 3, 0.3, 0.9), 3) if component in ("serving", "training") else None
    )
    return CostItem(
        component=component,
        startTime=start.isoformat(),
        endTime=end.isoformat(),
        environment=environment,
        project=project,
        namespace=namespace,
        cpuCost=round(_unit(seed, 0.4, 7.0) * scale, 4),
        memoryCost=round(_unit(seed + 1, 0.2, 4.0) * scale, 4),
        efficiency=round(_unit(seed + 2, 0.15, 0.95), 3),
        artifact=component,
        team=team,
        businessDomain=domain,
        usage=usage,
        gpuUtilization=gpu_utilization,
    )


def _ledger_items(
    namespace: str,
    environment: str,
    project: str | None,
    component: str | None,
    start: datetime,
    end: datetime,
) -> list[CostItem]:
    """Real AI-lifecycle cost recorded by the golden paths, mapped to the
    observer's CostItem shape. `cost_usd` lands in cpuCost (the frontend sums
    cpu+memory); stage/artifact/team/domain drive the stage/dimension views."""
    entries = cost_adapter.query_costs(
        start.isoformat(),
        end.isoformat(),
        namespace=namespace,
        environment=environment,
    )
    items: list[CostItem] = []
    for entry in entries:
        artifact = entry["artifact_id"]
        if component and artifact != component:
            continue
        if project and entry["team"] and entry["team"] != project and artifact != project:
            continue
        # Route the cost to the resource it was actually spent on, so the
        # frontend's AI cost model (cpu+memory+gpu+token) is populated rather
        # than everything landing in cpuCost. Token entries also carry usage,
        # which drives the unit-economics metric.
        unit = entry.get("unit", "")
        cost = entry["cost_usd"]
        is_gpu = unit == "gpu-hour"
        is_token = unit == "token"
        items.append(
            CostItem(
                component=artifact,
                startTime=entry["timestamp"],
                endTime=entry["timestamp"],
                environment=entry["environment"],
                project=entry["team"] or artifact,
                namespace=entry["namespace"],
                cpuCost=0.0 if (is_gpu or is_token) else cost,
                memoryCost=0.0,
                gpuCost=cost if is_gpu else 0.0,
                tokenCost=cost if is_token else 0.0,
                usage={"tokens": entry["quantity"]} if is_token else None,
                efficiency=1.0,
                stage=entry["stage"],
                team=entry["team"],
                businessDomain=entry["business_domain"],
                artifact=artifact,
            )
        )
    return items


@router.get(
    "/namespaces/{namespace}/environments/{environment}",
    response_model=CostItemsResponse,
)
def get_costs(
    namespace: str,
    environment: str,
    project: str | None = None,
    component: str | None = None,
    startTime: Annotated[str | None, Query()] = None,
    endTime: Annotated[str | None, Query()] = None,
    granularity: Annotated[str | None, Query()] = None,
) -> CostItemsResponse:
    now = datetime.now(UTC)
    start = _parse(startTime, now - timedelta(hours=24))
    end = _parse(endTime, now)
    targets = _targets(project, component)

    # Real AI cost from the ledger, on top of the infra baseline.
    ledger = _ledger_items(namespace, environment, project, component, start, end)

    # Real OpenCost allocation when configured; otherwise the synthetic baseline.
    # OpenCost rows are already window-aggregated, so they skip the bucketing.
    opencost = fetch_allocation(namespace, start, end)
    if opencost is not None:
        # to_cost_items returns plain dicts; pydantic coerces them to CostItem.
        return CostItemsResponse(
            items=ledger + cast(list[CostItem], to_cost_items(opencost, environment))
        )

    # No granularity: one item per target spanning the whole window.
    if not granularity:
        return CostItemsResponse(
            items=ledger + [_item(namespace, environment, p, c, start, end) for p, c in targets]
        )

    # With granularity: one item per target per time bucket, so the time-series
    # charts have a real series to stack.
    step = timedelta(hours=_GRANULARITY_HOURS.get(granularity, 1))
    buckets: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end and len(buckets) < 2000:
        bucket_end = min(cursor + step, end)
        buckets.append((cursor, bucket_end))
        cursor = bucket_end
    if not buckets:
        buckets = [(start, end)]

    items: list[CostItem] = list(ledger)
    for p, c in targets:
        for bucket_start, bucket_end in buckets:
            items.append(
                _item(
                    namespace,
                    environment,
                    p,
                    c,
                    bucket_start,
                    bucket_end,
                    scale=1.0 / len(buckets),
                )
            )
    return CostItemsResponse(items=items)


@router.get(
    "/namespaces/{namespace}/environments/{environment}/recommendations",
    response_model=CostRecommendationsResponse,
)
def get_cost_recommendations(
    namespace: str,
    environment: str,
    project: str | None = None,
    component: str | None = None,
    startTime: Annotated[str | None, Query()] = None,
    endTime: Annotated[str | None, Query()] = None,
) -> CostRecommendationsResponse:
    del startTime, endTime  # synthetic data doesn't depend on the window
    items: list[CostRecommendationItem] = []
    for p, c in _targets(project, component):
        seed = _seed(namespace, environment, p, c)
        cpu_req = 100 + seed % 400
        mem_req = 128 + seed % 512
        # Right-size to ~60% of the current request, so there's a visible saving.
        cpu_rec = max(10, int(cpu_req * 0.6))
        mem_rec = max(32, int(mem_req * 0.6))
        items.append(
            CostRecommendationItem(
                component=c,
                environment=environment,
                project=p,
                namespace=namespace,
                current=CostResourceProfile(
                    cpuRequest=f"{cpu_req}m",
                    cpuLimit=f"{cpu_req * 2}m",
                    memoryRequest=f"{mem_req}Mi",
                    memoryLimit=f"{mem_req * 2}Mi",
                    cpuCost=round(_unit(seed, 0.4, 7.0), 4),
                    memoryCost=round(_unit(seed + 1, 0.2, 4.0), 4),
                ),
                recommendation=CostResourceProfile(
                    cpuRequest=f"{cpu_rec}m",
                    cpuLimit=f"{cpu_rec * 2}m",
                    memoryRequest=f"{mem_rec}Mi",
                    memoryLimit=f"{mem_rec * 2}Mi",
                    cpuCost=round(_unit(seed, 0.4, 7.0) * 0.6, 4),
                    memoryCost=round(_unit(seed + 1, 0.2, 4.0) * 0.6, 4),
                ),
            )
        )
    return CostRecommendationsResponse(items=items)
