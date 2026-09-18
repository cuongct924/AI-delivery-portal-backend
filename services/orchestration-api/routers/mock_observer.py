"""Mock OpenChoreo observer for the Delivery Insights (DORA) API.

The Portal's Delivery Insights page calls the OpenChoreo observer directly
(`/api/v1alpha1/delivery-insights/dora/query` and `.../deployments/query`).
Without a real observability plane those calls have nothing to return, so this
router reproduces the two endpoints.

Data source is chosen by `MOCK_OBSERVER_DATA_SOURCE`:

- `mock` (default) — a deterministic synthetic generator seeded from the query
  scope, covering the whole requested window with plausible MLOps-template
  figures. Stable across refreshes, different per scope.
- `captured` — real Golden Path runs captured from the k3d cluster
  (`mock_data/delivery_insights_runs.json`, produced by
  `scripts/capture-delivery-insights-runs.sh`), used whenever the query window
  overlaps them and falling back to the synthetic generator otherwise.

Point the frontend's observer URL at orchestration-api (frontend config
`openchoreo.observability.mockObserverUrl`) and the page renders end to end.
"""

import hashlib
import json
import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, TypedDict

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1alpha1/delivery-insights", tags=["mock-observer"])

_REAL_RUNS_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "mock_data" / ("delivery_insights_runs.json")
)
_USE_CAPTURED_RUNS: Final[bool] = os.getenv("MOCK_OBSERVER_DATA_SOURCE", "mock").lower() in (
    "captured",
    "real",
)

# MLOps Golden Path templates (frontend repo `templates/`) — the components a
# deployment row can name when the query scope does not pin one.
_MLOPS_TEMPLATES: Final[tuple[str, ...]] = (
    "train-track-register",
    "register-deploy",
    "setup-model-monitoring",
)
_MLOPS_PROJECTS: Final[tuple[str, ...]] = (
    "fraud-detection",
    "churn-prediction",
    "customer-segmentation",
)
_ENVIRONMENTS: Final[tuple[str, ...]] = ("dev", "staging", "prod")
_DAYS_PER_BUCKET: Final[dict[str, int]] = {"daily": 1, "weekly": 7, "monthly": 30}
_OUTCOMES: Final[tuple[Literal["success", "failed", "in_progress"], ...]] = (
    "success",
    "failed",
    "in_progress",
)


class _CapturedStep(TypedDict):
    name: str
    started_at: str
    finished_at: str


class _CapturedRun(TypedDict):
    name: str
    started_at: str
    finished_at: str
    outcome: str
    steps: list[_CapturedStep]


def _load_real_runs() -> list[_CapturedRun]:
    """Loads the captured Golden Path runs, or [] when the file is absent."""
    try:
        raw = json.loads(_REAL_RUNS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    runs: list[_CapturedRun] = raw.get("runs", [])
    return [r for r in runs if r.get("started_at") and r.get("finished_at")]


class DoraSearchScope(BaseModel):
    namespace: str
    project: str | None = None
    component: str | None = None
    environment: str | None = None


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


class DoraMttrSummary(BaseModel):
    meanMs: int | None = None
    p50Ms: int | None = None
    recoveries: int
    classification: str
    deltaPct: float | None = None


class DoraSummary(BaseModel):
    deploymentFrequency: DoraFrequencySummary | None = None
    leadTime: DoraLeadTimeSummary | None = None
    changeFailureRate: DoraChangeFailureRateSummary | None = None
    mttr: DoraMttrSummary | None = None


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


class DoraSeries(BaseModel):
    deploymentFrequency: list[DoraFrequencyPoint] | None = None
    leadTime: list[DoraLeadTimePoint] | None = None
    changeFailureRate: list[DoraChangeFailureRatePoint] | None = None
    mttr: list[DoraMttrPoint] | None = None


class DoraDataAvailability(BaseModel):
    collecting: bool = True
    deliveryEvents: bool = True


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


class DoraDeploymentsResponse(BaseModel):
    deployments: list[DoraDeployment]
    totalCount: int
    tookMs: int


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _seed(scope: DoraSearchScope) -> int:
    key = "|".join(
        [scope.namespace, scope.project or "", scope.component or "", scope.environment or ""]
    )
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _bucket_starts(start: datetime, end: datetime, granularity: str) -> list[datetime]:
    step = timedelta(days=_DAYS_PER_BUCKET[granularity])
    buckets: list[datetime] = []
    cursor = start
    while cursor < end:
        buckets.append(cursor)
        cursor += step
    return buckets or [start]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def _classify_frequency(per_day: float) -> str:
    if per_day >= 1:
        return "Elite"
    if per_day >= 0.14:
        return "High"
    if per_day >= 0.03:
        return "Medium"
    return "Low"


def _classify_duration(ms: float | None) -> str:
    if ms is None:
        return "Unknown"
    hours = ms / 3_600_000
    if hours < 1:
        return "Elite"
    if hours < 24:
        return "High"
    if hours < 24 * 7:
        return "Medium"
    return "Low"


def _classify_rate(rate: float) -> str:
    if rate <= 0.15:
        return "Elite"
    if rate <= 0.30:
        return "High"
    if rate <= 0.45:
        return "Medium"
    return "Low"


def _build_metrics(request: DoraQueryRequest) -> DoraMetricsResponse:
    scope = request.searchScope
    start = _parse(request.startTime)
    end = _parse(request.endTime)
    granularity = request.granularity
    rng = random.Random(_seed(scope))
    window_days = max(1, (end - start).days)
    buckets = _bucket_starts(start, end, granularity)

    daily_rate = rng.uniform(0.4, 1.6)
    failure_rate = rng.uniform(0.05, 0.28)
    lead_mu = rng.uniform(1.2, 2.2)
    mttr_mu = rng.uniform(0.8, 1.8)

    freq_points: list[DoraFrequencyPoint] = []
    cfr_points: list[DoraChangeFailureRatePoint] = []
    lead_points: list[DoraLeadTimePoint] = []
    mttr_points: list[DoraMttrPoint] = []
    all_lead_ms: list[float] = []
    all_mttr_ms: list[float] = []
    total_deployments = 0
    total_failed = 0

    for bucket in buckets:
        days = _DAYS_PER_BUCKET[granularity]
        expected = daily_rate * days
        count = max(0, round(rng.gauss(expected, max(0.5, expected**0.5))))
        failed = sum(1 for _ in range(count) if rng.random() < failure_rate)
        total_deployments += count
        total_failed += failed

        freq_points.append(DoraFrequencyPoint(bucketStart=bucket.isoformat(), count=count))
        cfr_points.append(
            DoraChangeFailureRatePoint(
                bucketStart=bucket.isoformat(),
                rate=(failed / count) if count else 0.0,
                failed=failed,
                total=count,
            )
        )

        bucket_lead = [rng.lognormvariate(lead_mu, 0.7) * 3_600_000 for _ in range(count - failed)]
        if bucket_lead:
            all_lead_ms.extend(bucket_lead)
            lead_points.append(
                DoraLeadTimePoint(
                    bucketStart=bucket.isoformat(),
                    p50Ms=round(_percentile(bucket_lead, 0.5)),
                    p75Ms=round(_percentile(bucket_lead, 0.75)),
                    p95Ms=round(_percentile(bucket_lead, 0.95)),
                )
            )

        bucket_mttr = [rng.lognormvariate(mttr_mu, 0.8) * 3_600_000 for _ in range(failed)]
        if bucket_mttr:
            all_mttr_ms.extend(bucket_mttr)
            mttr_points.append(
                DoraMttrPoint(
                    bucketStart=bucket.isoformat(),
                    meanMs=round(sum(bucket_mttr) / len(bucket_mttr)),
                    p50Ms=round(_percentile(bucket_mttr, 0.5)),
                    count=len(bucket_mttr),
                )
            )

    per_day = total_deployments / window_days
    cfr_rate = (total_failed / total_deployments) if total_deployments else 0.0
    lead_p50 = _percentile(all_lead_ms, 0.5) if all_lead_ms else None
    lead_p95 = _percentile(all_lead_ms, 0.95) if all_lead_ms else None
    mttr_mean = (sum(all_mttr_ms) / len(all_mttr_ms)) if all_mttr_ms else None
    mttr_p50 = _percentile(all_mttr_ms, 0.5) if all_mttr_ms else None

    return DoraMetricsResponse(
        dataAvailability=DoraDataAvailability(),
        scope=scope,
        granularity=granularity,
        window=DoraWindow(
            startTime=start.isoformat(),
            endTime=end.isoformat(),
            generatedAt=datetime.now(UTC).isoformat(),
        ),
        summary=DoraSummary(
            deploymentFrequency=DoraFrequencySummary(
                total=total_deployments,
                perDay=round(per_day, 2),
                classification=_classify_frequency(per_day),
                deltaPct=round(rng.uniform(-25, 25), 1),
            ),
            leadTime=DoraLeadTimeSummary(
                p50Ms=round(lead_p50) if lead_p50 is not None else None,
                p95Ms=round(lead_p95) if lead_p95 is not None else None,
                coverage=round(rng.uniform(0.7, 1.0), 2),
                classification=_classify_duration(lead_p50),
                deltaPct=round(rng.uniform(-20, 20), 1),
            ),
            changeFailureRate=DoraChangeFailureRateSummary(
                rate=round(cfr_rate, 4),
                failed=total_failed,
                total=total_deployments,
                classification=_classify_rate(cfr_rate),
                deltaPct=round(rng.uniform(-20, 20), 1),
            ),
            mttr=DoraMttrSummary(
                meanMs=round(mttr_mean) if mttr_mean is not None else None,
                p50Ms=round(mttr_p50) if mttr_p50 is not None else None,
                recoveries=len(all_mttr_ms),
                classification=_classify_duration(mttr_mean),
                deltaPct=round(rng.uniform(-20, 20), 1),
            ),
        ),
        series=DoraSeries(
            deploymentFrequency=freq_points,
            leadTime=lead_points,
            changeFailureRate=cfr_points,
            mttr=mttr_points,
        ),
    )


def _build_metrics_from_runs(
    runs: list[_CapturedRun], request: DoraQueryRequest
) -> DoraMetricsResponse:
    """Aggregates captured Golden Path runs into a DORA response.

    A run is a "change": success = deployment, failure = failed change. Lead
    time is the run's wall-clock duration; MTTR is the gap from a failure to
    the next success. Buckets with no runs stay zero-filled for frequency and
    failure rate, and are omitted for lead time / MTTR (matching the real
    observer's sparse-series contract).
    """
    scope = request.searchScope
    start = _parse(request.startTime)
    end = _parse(request.endTime)
    granularity = request.granularity
    rng = random.Random(_seed(scope))
    window_days = max(1, (end - start).days)
    buckets = _bucket_starts(start, end, granularity)
    bucket_starts = [b.isoformat() for b in buckets]

    in_window = [r for r in runs if start <= _parse(r["started_at"]) < end]
    in_window.sort(key=lambda r: r["started_at"])

    freq_counts: dict[str, int] = dict.fromkeys(bucket_starts, 0)
    cfr_failed: dict[str, int] = dict.fromkeys(bucket_starts, 0)
    cfr_total: dict[str, int] = dict.fromkeys(bucket_starts, 0)
    lead_by_bucket: dict[str, list[float]] = {b: [] for b in bucket_starts}
    mttr_by_bucket: dict[str, list[float]] = {b: [] for b in bucket_starts}

    def _bucket_for(ts: datetime) -> str:
        chosen = bucket_starts[0]
        for bucket in buckets:
            if bucket <= ts:
                chosen = bucket.isoformat()
            else:
                break
        return chosen

    all_lead_ms: list[float] = []
    all_mttr_ms: list[float] = []
    total_changes = 0
    total_failed = 0
    total_success = 0

    for run in in_window:
        started = _parse(run["started_at"])
        finished = _parse(run["finished_at"])
        bucket = _bucket_for(started)
        duration_ms = (finished - started).total_seconds() * 1000
        total_changes += 1
        cfr_total[bucket] += 1
        if run["outcome"] == "success":
            total_success += 1
            freq_counts[bucket] += 1
            lead_by_bucket[bucket].append(duration_ms)
            all_lead_ms.append(duration_ms)
        else:
            total_failed += 1
            cfr_failed[bucket] += 1

    # MTTR: gap from each failure to the next success.
    for index, run in enumerate(in_window):
        if run["outcome"] != "failure":
            continue
        failure_end = _parse(run["finished_at"])
        for later in in_window[index + 1 :]:
            if later["outcome"] == "success":
                recovery_ms = (_parse(later["started_at"]) - failure_end).total_seconds() * 1000
                if recovery_ms >= 0:
                    bucket = _bucket_for(failure_end)
                    mttr_by_bucket[bucket].append(recovery_ms)
                    all_mttr_ms.append(recovery_ms)
                break

    per_day = total_success / window_days
    cfr_rate = (total_failed / total_changes) if total_changes else 0.0
    lead_p50 = _percentile(all_lead_ms, 0.5) if all_lead_ms else None
    lead_p95 = _percentile(all_lead_ms, 0.95) if all_lead_ms else None
    mttr_mean = (sum(all_mttr_ms) / len(all_mttr_ms)) if all_mttr_ms else None
    mttr_p50 = _percentile(all_mttr_ms, 0.5) if all_mttr_ms else None

    freq_points = [DoraFrequencyPoint(bucketStart=b, count=freq_counts[b]) for b in bucket_starts]
    cfr_points = [
        DoraChangeFailureRatePoint(
            bucketStart=b,
            rate=(cfr_failed[b] / cfr_total[b]) if cfr_total[b] else 0.0,
            failed=cfr_failed[b],
            total=cfr_total[b],
        )
        for b in bucket_starts
    ]
    lead_points = [
        DoraLeadTimePoint(
            bucketStart=b,
            p50Ms=round(_percentile(lead_by_bucket[b], 0.5)),
            p75Ms=round(_percentile(lead_by_bucket[b], 0.75)),
            p95Ms=round(_percentile(lead_by_bucket[b], 0.95)),
        )
        for b in bucket_starts
        if lead_by_bucket[b]
    ]
    mttr_points = [
        DoraMttrPoint(
            bucketStart=b,
            meanMs=round(sum(mttr_by_bucket[b]) / len(mttr_by_bucket[b])),
            p50Ms=round(_percentile(mttr_by_bucket[b], 0.5)),
            count=len(mttr_by_bucket[b]),
        )
        for b in bucket_starts
        if mttr_by_bucket[b]
    ]

    return DoraMetricsResponse(
        dataAvailability=DoraDataAvailability(),
        scope=scope,
        granularity=granularity,
        window=DoraWindow(
            startTime=start.isoformat(),
            endTime=end.isoformat(),
            generatedAt=datetime.now(UTC).isoformat(),
        ),
        summary=DoraSummary(
            deploymentFrequency=DoraFrequencySummary(
                total=total_success,
                perDay=round(per_day, 2),
                classification=_classify_frequency(per_day),
                deltaPct=round(rng.uniform(-25, 25), 1),
            ),
            leadTime=DoraLeadTimeSummary(
                p50Ms=round(lead_p50) if lead_p50 is not None else None,
                p95Ms=round(lead_p95) if lead_p95 is not None else None,
                coverage=1.0,
                classification=_classify_duration(lead_p50),
                deltaPct=round(rng.uniform(-20, 20), 1),
            ),
            changeFailureRate=DoraChangeFailureRateSummary(
                rate=round(cfr_rate, 4),
                failed=total_failed,
                total=total_changes,
                classification=_classify_rate(cfr_rate),
                deltaPct=round(rng.uniform(-20, 20), 1),
            ),
            mttr=DoraMttrSummary(
                meanMs=round(mttr_mean) if mttr_mean is not None else None,
                p50Ms=round(mttr_p50) if mttr_p50 is not None else None,
                recoveries=len(all_mttr_ms),
                classification=_classify_duration(mttr_mean),
                deltaPct=round(rng.uniform(-20, 20), 1),
            ),
        ),
        series=DoraSeries(
            deploymentFrequency=freq_points,
            leadTime=lead_points,
            changeFailureRate=cfr_points,
            mttr=mttr_points,
        ),
    )


def _apply_metric_filter(response: DoraMetricsResponse, metrics: list[str] | None) -> None:
    if not metrics:
        return
    wanted = set(metrics)
    if "deploymentFrequency" not in wanted:
        response.summary.deploymentFrequency = None
        response.series.deploymentFrequency = None
    if "leadTime" not in wanted:
        response.summary.leadTime = None
        response.series.leadTime = None
    if "changeFailureRate" not in wanted:
        response.summary.changeFailureRate = None
        response.series.changeFailureRate = None
    if "mttr" not in wanted:
        response.summary.mttr = None
        response.series.mttr = None


def _build_deployments_from_runs(
    runs: list[_CapturedRun], request: DoraDeploymentsQueryRequest
) -> DoraDeploymentsResponse:
    """Maps captured Golden Path runs to deployment rows."""
    scope = request.searchScope
    start = _parse(request.startTime)
    end = _parse(request.endTime)
    in_window = [r for r in runs if start <= _parse(r["started_at"]) < end]
    in_window.sort(key=lambda r: r["started_at"], reverse=request.sortOrder == "desc")

    deployments: list[DoraDeployment] = []
    for run in in_window[: request.limit]:
        started = _parse(run["started_at"])
        finished = _parse(run["finished_at"])
        failed = run["outcome"] != "success"
        deployments.append(
            DoraDeployment(
                deployedAt=finished.isoformat(),
                projectName=scope.project or "telco-fraud-detection",
                componentName=scope.component or "train-track-register",
                environmentName=scope.environment or "development",
                componentRelease=run["name"],
                commit="",
                outcome="failed" if failed else "success",
                failedBy="train-step" if failed else "",
                failureReason="training failed" if failed else "",
                incidentId=f"INC-{run['name'][-6:]}" if failed else "",
                leadTimeMs=round((finished - started).total_seconds() * 1000),
            )
        )
    return DoraDeploymentsResponse(deployments=deployments, totalCount=len(deployments), tookMs=1)


def _build_deployments(request: DoraDeploymentsQueryRequest) -> DoraDeploymentsResponse:
    scope = request.searchScope
    start = _parse(request.startTime)
    end = _parse(request.endTime)
    rng = random.Random(_seed(scope) ^ 0x5EED)
    span = max(1.0, (end - start).total_seconds())
    count = min(request.limit, rng.randint(8, 40))

    deployments: list[DoraDeployment] = []
    for _ in range(count):
        deployed_at = start + timedelta(seconds=rng.random() * span)
        outcome = rng.choice(_OUTCOMES)
        failed = outcome == "failed"
        component = scope.component or rng.choice(_MLOPS_TEMPLATES)
        project = scope.project or rng.choice(_MLOPS_PROJECTS)
        environment = scope.environment or rng.choice(_ENVIRONMENTS)
        deployments.append(
            DoraDeployment(
                deployedAt=deployed_at.isoformat(),
                projectName=project,
                componentName=component,
                environmentName=environment,
                componentRelease=f"{component}-{rng.randint(1, 40)}",
                commit="".join(rng.choice("0123456789abcdef") for _ in range(40)),
                outcome=outcome,
                failedBy="evaluate-gate" if failed else "",
                failureReason="eval score below threshold" if failed else "",
                incidentId=f"INC-{rng.randint(1000, 9999)}" if failed else "",
                leadTimeMs=round(rng.lognormvariate(1.6, 0.7) * 3_600_000),
            )
        )

    deployments.sort(key=lambda d: d.deployedAt, reverse=request.sortOrder == "desc")
    return DoraDeploymentsResponse(
        deployments=deployments,
        totalCount=len(deployments),
        tookMs=rng.randint(3, 25),
    )


@router.post("/dora/query", response_model=DoraMetricsResponse)
def query_dora_metrics(request: DoraQueryRequest) -> DoraMetricsResponse:
    """Returns DORA metrics for the requested scope and window.

    Serves captured real Golden Path runs when `MOCK_OBSERVER_DATA_SOURCE=captured`
    and the window overlaps them, and the synthetic generator otherwise.
    """
    runs = _load_real_runs() if _USE_CAPTURED_RUNS else []
    start = _parse(request.startTime)
    end = _parse(request.endTime)
    has_real = any(start <= _parse(run["started_at"]) < end for run in runs)
    response = _build_metrics_from_runs(runs, request) if has_real else _build_metrics(request)
    _apply_metric_filter(response, request.metrics)
    return response


@router.post("/dora/deployments/query", response_model=DoraDeploymentsResponse)
def query_dora_deployments(request: DoraDeploymentsQueryRequest) -> DoraDeploymentsResponse:
    """Returns deployment events for the requested scope and window.

    Serves captured real Golden Path runs when `MOCK_OBSERVER_DATA_SOURCE=captured`
    and the window overlaps them, and the synthetic generator otherwise.
    """
    runs = _load_real_runs() if _USE_CAPTURED_RUNS else []
    start = _parse(request.startTime)
    end = _parse(request.endTime)
    has_real = any(start <= _parse(run["started_at"]) < end for run in runs)
    if has_real:
        return _build_deployments_from_runs(runs, request)
    return _build_deployments(request)
