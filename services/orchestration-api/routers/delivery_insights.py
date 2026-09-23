"""Delivery Insights (DORA) API for the Portal dashboard.

The Portal's Delivery Insights page calls the OpenChoreo observer directly
(`/api/v1alpha1/delivery-insights/dora/query` and `.../deployments/query`).
This router reproduces that contract and delegates to
`adapters.factory.get_delivery_observer_adapter()` — mock synthetic/captured
data by default, or OpenChoreo's real Prometheus + MLflow when
`USE_MOCK_DELIVERY_OBSERVER=false` (see adapters/delivery/interfaces.py's
IDeliveryObserverAdapter).

Point the frontend's observer URL at orchestration-api (frontend config
`openchoreo.observability.mockObserverUrl`) and the page renders end to end.
"""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from adapters.delivery.interfaces import DeliveryScope
from adapters.factory import get_delivery_observer_adapter

router = APIRouter(prefix="/api/v1alpha1/delivery-insights", tags=["delivery-insights"])

delivery_observer_adapter = get_delivery_observer_adapter()


class DoraSearchScope(BaseModel):
    namespace: str
    project: str | None = None
    component: str | None = None
    environment: str | None = None
    workloadType: Literal["service", "ml_model", "llm_app"] | None = None


class DoraQueryRequest(BaseModel):
    searchScope: DoraSearchScope
    startTime: str
    endTime: str
    granularity: Literal["daily", "weekly", "monthly"] = "daily"
    metrics: list[str] | None = None


class DoraDeploymentsQueryRequest(BaseModel):
    searchScope: DoraSearchScope
    startTime: str
    endTime: str
    limit: int = 100
    sortOrder: Literal["asc", "desc"] = "desc"


class DoraFrequencySummary(BaseModel):
    total: int
    perDay: float
    classification: str
    deltaPct: float | None = None


class DoraLeadTimeSummary(BaseModel):
    p50Ms: int | None = None
    p95Ms: int | None = None
    coverage: float
    classification: str
    deltaPct: float | None = None


class DoraChangeFailureRateSummary(BaseModel):
    rate: float
    failed: int
    total: int
    classification: str
    deltaPct: float | None = None
    infraCfr: float | None = None
    semanticCfr: float | None = None


class DoraMttrSummary(BaseModel):
    meanMs: int | None = None
    p50Ms: int | None = None
    recoveries: int
    classification: str
    deltaPct: float | None = None


class DoraReworkRateSummary(BaseModel):
    rate: float
    reworked: int
    total: int
    classification: str
    deltaPct: float | None = None


class DoraSummary(BaseModel):
    deploymentFrequency: DoraFrequencySummary | None = None
    leadTime: DoraLeadTimeSummary | None = None
    changeFailureRate: DoraChangeFailureRateSummary | None = None
    mttr: DoraMttrSummary | None = None
    reworkRate: DoraReworkRateSummary | None = None


class DoraFrequencyPoint(BaseModel):
    bucketStart: str
    count: int


class DoraLeadTimePoint(BaseModel):
    bucketStart: str
    p50Ms: int
    p75Ms: int
    p95Ms: int


class DoraChangeFailureRatePoint(BaseModel):
    bucketStart: str
    rate: float
    failed: int
    total: int


class DoraMttrPoint(BaseModel):
    bucketStart: str
    meanMs: int
    p50Ms: int
    count: int


class DoraReworkRatePoint(BaseModel):
    bucketStart: str
    rate: float
    reworked: int
    total: int


class DoraSeries(BaseModel):
    deploymentFrequency: list[DoraFrequencyPoint] | None = None
    leadTime: list[DoraLeadTimePoint] | None = None
    changeFailureRate: list[DoraChangeFailureRatePoint] | None = None
    mttr: list[DoraMttrPoint] | None = None
    reworkRate: list[DoraReworkRatePoint] | None = None


class DoraDataAvailability(BaseModel):
    collecting: bool = True
    deliveryEvents: bool = True
    evalPipeline: bool = False
    driftMonitor: bool = False
    guardrails: bool = False


class DoraWindow(BaseModel):
    startTime: str
    endTime: str
    generatedAt: str


class DoraMetricsResponse(BaseModel):
    dataAvailability: DoraDataAvailability
    scope: DoraSearchScope
    granularity: Literal["daily", "weekly", "monthly"]
    window: DoraWindow
    summary: DoraSummary
    series: DoraSeries


class DoraDeployment(BaseModel):
    deployedAt: str
    projectName: str
    componentName: str
    environmentName: str
    componentRelease: str
    commit: str
    outcome: Literal["success", "failed", "in_progress"]
    failedBy: str
    failureReason: str
    incidentId: str
    leadTimeMs: int | None = None
    workloadType: Literal["service", "ml_model", "llm_app"] | None = None
    changeType: Literal["infra", "model", "rag_index", "prompt"] | None = None
    driftTriggered: bool = False
    evalCoverage: float | None = None
    leadTimeBreakdown: dict[str, int] | None = None
    evalBottleneck: Literal["data_prep", "train", "eval", "deploy"] | None = None
    failureClass: Literal["infra", "semantic"] | None = None
    semanticType: (
        Literal["accuracy_drop", "drift", "hallucination", "guardrail", "prompt_injection"] | None
    ) = None
    evalScore: float | None = None
    baselineScore: float | None = None
    driftScore: float | None = None
    recoveryStrategy: Literal["rollback", "fallback", "guardrail", "retrain"] | None = None
    modelVersion: str | None = None
    promptVersion: str | None = None
    ragIndexVersion: str | None = None


class DoraDeploymentsResponse(BaseModel):
    deployments: list[DoraDeployment]
    totalCount: int
    tookMs: int


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@router.post("/dora/query", response_model=DoraMetricsResponse)
def query_dora_metrics(request: DoraQueryRequest) -> DoraMetricsResponse:
    result = delivery_observer_adapter.query_metrics(
        scope=DeliveryScope(**request.searchScope.model_dump()),
        start=_parse(request.startTime),
        end=_parse(request.endTime),
        granularity=request.granularity,
        metrics=request.metrics,
    )
    return DoraMetricsResponse.model_validate(result)


@router.post("/dora/deployments/query", response_model=DoraDeploymentsResponse)
def query_dora_deployments(request: DoraDeploymentsQueryRequest) -> DoraDeploymentsResponse:
    result = delivery_observer_adapter.query_deployments(
        scope=DeliveryScope(**request.searchScope.model_dump()),
        start=_parse(request.startTime),
        end=_parse(request.endTime),
        limit=request.limit,
        sort_order=request.sortOrder,
    )
    return DoraDeploymentsResponse.model_validate(result)
