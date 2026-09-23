"""Mock IDeliveryObserverAdapter — deterministic synthetic DORA data (or,
when MOCK_OBSERVER_DATA_SOURCE=captured, real captured Golden Path runs),
extended with synthetic ML/LLM dimensions so local/demo mode never shows an
all-None dashboard. Ported out of the old routers/mock_observer.py so the
router can stay adapter-agnostic (see IDeliveryObserverAdapter).
"""

import hashlib
import os
import random
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

from adapters.delivery._captured_runs import (
    CapturedRun,
    build_deployment_rows,
    load_captured_runs,
    parse_iso,
)
from adapters.delivery.interfaces import (
    ChangeType,
    DeliveryAvailability,
    DeliveryChangeFailureRatePoint,
    DeliveryChangeFailureRateSummary,
    DeliveryDeployment,
    DeliveryDeploymentsResult,
    DeliveryFrequencyPoint,
    DeliveryFrequencySummary,
    DeliveryLeadTimePoint,
    DeliveryLeadTimeSummary,
    DeliveryMetricsResult,
    DeliveryMttrPoint,
    DeliveryMttrSummary,
    DeliveryReworkRatePoint,
    DeliveryReworkRateSummary,
    DeliveryScope,
    DeliverySeries,
    DeliverySummary,
    DeliveryWindow,
    DoraGranularity,
    IDeliveryObserverAdapter,
    LifecyclePhase,
    RecoveryStrategy,
    SemanticFailureType,
    workload_type_for,
)

# MLOps Golden Path templates — fallback component names when scope doesn't pin one.
_MLOPS_TEMPLATES: Final[tuple[str, ...]] = (
    "train-track-register",
    "evaluate-deploy-model",
    "setup-model-monitoring",
)
_MLOPS_PROJECTS: Final[tuple[str, ...]] = (
    "telco-fraud-detection",
    "customer-support-ai",
    "churn-prediction",
)
_ENVIRONMENTS: Final[tuple[str, ...]] = ("dev", "staging", "prod")
_DAYS_PER_BUCKET: Final[dict[str, int]] = {"daily": 1, "weekly": 7, "monthly": 30}
_OUTCOMES: Final[tuple[Literal["success", "failed", "in_progress"], ...]] = (
    "success",
    "failed",
    "in_progress",
)
_CHANGE_TYPES: Final[tuple[ChangeType, ...]] = ("infra", "model", "rag_index", "prompt")
_LIFECYCLE_PHASES: Final[tuple[LifecyclePhase, ...]] = ("data_prep", "train", "eval", "deploy")
_SEMANTIC_TYPES: Final[tuple[SemanticFailureType, ...]] = (
    "accuracy_drop",
    "drift",
    "hallucination",
    "guardrail",
    "prompt_injection",
)
_RECOVERY_STRATEGIES: Final[tuple[RecoveryStrategy, ...]] = (
    "rollback",
    "fallback",
    "guardrail",
    "retrain",
)
_SEMANTIC_FAILURE_SHARE: Final = 0.6


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


_CHANGE_TYPES_BY_WORKLOAD: Final[dict[str, tuple[ChangeType, ...]]] = {
    "service": ("infra",),
    "ml_model": ("model",),
    "llm_app": ("rag_index", "prompt"),
}


def _change_types_for(workload_type: str | None) -> tuple[ChangeType, ...]:
    if workload_type is None:
        return _CHANGE_TYPES
    return _CHANGE_TYPES_BY_WORKLOAD[workload_type]


def _seed(scope: DeliveryScope) -> int:
    key = "|".join(
        [
            scope["namespace"],
            scope["project"] or "",
            scope["component"] or "",
            scope["environment"] or "",
        ]
    )
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _synthetic_lead_time_breakdown(
    rng: random.Random, total_ms: float
) -> dict[LifecyclePhase, int]:
    weights = [rng.uniform(0.5, 1.5) for _ in _LIFECYCLE_PHASES]
    weight_sum = sum(weights)
    return {
        phase: round(total_ms * weight / weight_sum)
        for phase, weight in zip(_LIFECYCLE_PHASES, weights, strict=True)
    }


class MockDeliveryObserverAdapter(IDeliveryObserverAdapter):
    """Synthetic (or captured-JSON) DORA data — the default backend until
    `USE_MOCK_DELIVERY_OBSERVER=false` switches to
    PrometheusDeliveryObserverAdapter."""

    def __init__(self, data_source: str | None = None) -> None:
        source = (data_source or os.getenv("MOCK_OBSERVER_DATA_SOURCE", "mock")).lower()
        self._use_captured_runs = source in ("captured", "real")

    def query_metrics(
        self,
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        granularity: DoraGranularity,
        metrics: list[str] | None,
    ) -> DeliveryMetricsResult:
        runs = load_captured_runs() if self._use_captured_runs else []
        has_real = any(start <= parse_iso(run["started_at"]) < end for run in runs)
        result = (
            self._build_metrics_from_runs(runs, scope, start, end, granularity)
            if has_real
            else self._build_metrics(scope, start, end, granularity)
        )
        self._apply_metric_filter(result, metrics)
        return result

    def query_deployments(
        self,
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        limit: int,
        sort_order: Literal["asc", "desc"],
    ) -> DeliveryDeploymentsResult:
        runs = load_captured_runs() if self._use_captured_runs else []
        has_real = any(start <= parse_iso(run["started_at"]) < end for run in runs)
        if has_real:
            rows = build_deployment_rows(runs, scope, start, end, limit, sort_order)
            return DeliveryDeploymentsResult(deployments=rows, totalCount=len(rows), tookMs=1)
        return self._build_deployments(scope, start, end, limit, sort_order)

    def _build_metrics(
        self, scope: DeliveryScope, start: datetime, end: datetime, granularity: DoraGranularity
    ) -> DeliveryMetricsResult:
        rng = random.Random(_seed(scope))
        window_days = max(1, (end - start).days)
        buckets = _bucket_starts(start, end, granularity)

        daily_rate = rng.uniform(0.4, 1.6)
        failure_rate = rng.uniform(0.05, 0.28)
        lead_mu = rng.uniform(1.2, 2.2)
        mttr_mu = rng.uniform(0.8, 1.8)
        rework_rate = rng.uniform(0.05, 0.25)

        freq_points: list[DeliveryFrequencyPoint] = []
        cfr_points: list[DeliveryChangeFailureRatePoint] = []
        lead_points: list[DeliveryLeadTimePoint] = []
        mttr_points: list[DeliveryMttrPoint] = []
        rework_points: list[DeliveryReworkRatePoint] = []
        all_lead_ms: list[float] = []
        all_mttr_ms: list[float] = []
        total_deployments = 0
        total_failed = 0
        total_semantic_failed = 0
        total_reworked = 0

        for bucket in buckets:
            days = _DAYS_PER_BUCKET[granularity]
            expected = daily_rate * days
            count = max(0, round(rng.gauss(expected, max(0.5, expected**0.5))))
            failed = sum(1 for _ in range(count) if rng.random() < failure_rate)
            semantic_failed = sum(
                1 for _ in range(failed) if rng.random() < _SEMANTIC_FAILURE_SHARE
            )
            total_deployments += count
            total_failed += failed
            total_semantic_failed += semantic_failed

            freq_points.append(DeliveryFrequencyPoint(bucketStart=bucket.isoformat(), count=count))
            cfr_points.append(
                DeliveryChangeFailureRatePoint(
                    bucketStart=bucket.isoformat(),
                    rate=(failed / count) if count else 0.0,
                    failed=failed,
                    total=count,
                )
            )

            reworked = sum(1 for _ in range(count) if rng.random() < rework_rate)
            total_reworked += reworked
            rework_points.append(
                DeliveryReworkRatePoint(
                    bucketStart=bucket.isoformat(),
                    rate=(reworked / count) if count else 0.0,
                    reworked=reworked,
                    total=count,
                )
            )

            bucket_lead = [
                rng.lognormvariate(lead_mu, 0.7) * 3_600_000 for _ in range(count - failed)
            ]
            if bucket_lead:
                all_lead_ms.extend(bucket_lead)
                lead_points.append(
                    DeliveryLeadTimePoint(
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
                    DeliveryMttrPoint(
                        bucketStart=bucket.isoformat(),
                        meanMs=round(sum(bucket_mttr) / len(bucket_mttr)),
                        p50Ms=round(_percentile(bucket_mttr, 0.5)),
                        count=len(bucket_mttr),
                    )
                )

        per_day = total_deployments / window_days
        cfr_rate = (total_failed / total_deployments) if total_deployments else 0.0
        infra_cfr = (
            ((total_failed - total_semantic_failed) / total_deployments)
            if total_deployments
            else 0.0
        )
        semantic_cfr = (total_semantic_failed / total_deployments) if total_deployments else 0.0
        lead_p50 = _percentile(all_lead_ms, 0.5) if all_lead_ms else None
        lead_p95 = _percentile(all_lead_ms, 0.95) if all_lead_ms else None
        mttr_mean = (sum(all_mttr_ms) / len(all_mttr_ms)) if all_mttr_ms else None
        mttr_p50 = _percentile(all_mttr_ms, 0.5) if all_mttr_ms else None
        overall_rework_rate = (total_reworked / total_deployments) if total_deployments else 0.0

        return DeliveryMetricsResult(
            dataAvailability=DeliveryAvailability(
                collecting=True,
                deliveryEvents=True,
                evalPipeline=True,
                driftMonitor=True,
                guardrails=False,
            ),
            scope=scope,
            granularity=granularity,
            window=DeliveryWindow(
                startTime=start.isoformat(),
                endTime=end.isoformat(),
                generatedAt=datetime.now(UTC).isoformat(),
            ),
            summary=DeliverySummary(
                deploymentFrequency=DeliveryFrequencySummary(
                    total=total_deployments,
                    perDay=round(per_day, 2),
                    classification=_classify_frequency(per_day),
                    deltaPct=round(rng.uniform(-25, 25), 1),
                ),
                leadTime=DeliveryLeadTimeSummary(
                    p50Ms=round(lead_p50) if lead_p50 is not None else None,
                    p95Ms=round(lead_p95) if lead_p95 is not None else None,
                    coverage=round(rng.uniform(0.7, 1.0), 2),
                    classification=_classify_duration(lead_p50),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                ),
                changeFailureRate=DeliveryChangeFailureRateSummary(
                    rate=round(cfr_rate, 4),
                    failed=total_failed,
                    total=total_deployments,
                    classification=_classify_rate(cfr_rate),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                    infraCfr=round(infra_cfr, 4),
                    semanticCfr=round(semantic_cfr, 4),
                ),
                mttr=DeliveryMttrSummary(
                    meanMs=round(mttr_mean) if mttr_mean is not None else None,
                    p50Ms=round(mttr_p50) if mttr_p50 is not None else None,
                    recoveries=len(all_mttr_ms),
                    classification=_classify_duration(mttr_mean),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                ),
                reworkRate=DeliveryReworkRateSummary(
                    rate=round(overall_rework_rate, 4),
                    reworked=total_reworked,
                    total=total_deployments,
                    classification=_classify_rate(overall_rework_rate),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                ),
            ),
            series=DeliverySeries(
                deploymentFrequency=freq_points,
                leadTime=lead_points,
                changeFailureRate=cfr_points,
                mttr=mttr_points,
                reworkRate=rework_points,
            ),
        )

    def _build_metrics_from_runs(
        self,
        runs: list[CapturedRun],
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        granularity: DoraGranularity,
    ) -> DeliveryMetricsResult:
        """Aggregates captured Golden Path runs into a DORA response.

        A run is a "change": success = deployment, failure = failed change.
        Lead time is the run's wall-clock duration; MTTR is the gap from a
        failure to the next success. The captured JSON carries no
        eval/drift tags, so infraCfr/semanticCfr stay None here rather than
        fabricating a split the data can't support.
        """
        rng = random.Random(_seed(scope))
        window_days = max(1, (end - start).days)
        buckets = _bucket_starts(start, end, granularity)
        bucket_starts = [b.isoformat() for b in buckets]

        in_window = [r for r in runs if start <= parse_iso(r["started_at"]) < end]
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
            started = parse_iso(run["started_at"])
            finished = parse_iso(run["finished_at"])
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

        for index, run in enumerate(in_window):
            if run["outcome"] != "failure":
                continue
            failure_end = parse_iso(run["finished_at"])
            for later in in_window[index + 1 :]:
                if later["outcome"] == "success":
                    recovery_ms = (
                        parse_iso(later["started_at"]) - failure_end
                    ).total_seconds() * 1000
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

        freq_points = [
            DeliveryFrequencyPoint(bucketStart=b, count=freq_counts[b]) for b in bucket_starts
        ]
        cfr_points = [
            DeliveryChangeFailureRatePoint(
                bucketStart=b,
                rate=(cfr_failed[b] / cfr_total[b]) if cfr_total[b] else 0.0,
                failed=cfr_failed[b],
                total=cfr_total[b],
            )
            for b in bucket_starts
        ]
        lead_points = [
            DeliveryLeadTimePoint(
                bucketStart=b,
                p50Ms=round(_percentile(lead_by_bucket[b], 0.5)),
                p75Ms=round(_percentile(lead_by_bucket[b], 0.75)),
                p95Ms=round(_percentile(lead_by_bucket[b], 0.95)),
            )
            for b in bucket_starts
            if lead_by_bucket[b]
        ]
        mttr_points = [
            DeliveryMttrPoint(
                bucketStart=b,
                meanMs=round(sum(mttr_by_bucket[b]) / len(mttr_by_bucket[b])),
                p50Ms=round(_percentile(mttr_by_bucket[b], 0.5)),
                count=len(mttr_by_bucket[b]),
            )
            for b in bucket_starts
            if mttr_by_bucket[b]
        ]

        return DeliveryMetricsResult(
            dataAvailability=DeliveryAvailability(
                collecting=True,
                deliveryEvents=True,
                evalPipeline=False,
                driftMonitor=False,
                guardrails=False,
            ),
            scope=scope,
            granularity=granularity,
            window=DeliveryWindow(
                startTime=start.isoformat(),
                endTime=end.isoformat(),
                generatedAt=datetime.now(UTC).isoformat(),
            ),
            summary=DeliverySummary(
                deploymentFrequency=DeliveryFrequencySummary(
                    total=total_success,
                    perDay=round(per_day, 2),
                    classification=_classify_frequency(per_day),
                    deltaPct=round(rng.uniform(-25, 25), 1),
                ),
                leadTime=DeliveryLeadTimeSummary(
                    p50Ms=round(lead_p50) if lead_p50 is not None else None,
                    p95Ms=round(lead_p95) if lead_p95 is not None else None,
                    coverage=1.0,
                    classification=_classify_duration(lead_p50),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                ),
                changeFailureRate=DeliveryChangeFailureRateSummary(
                    rate=round(cfr_rate, 4),
                    failed=total_failed,
                    total=total_changes,
                    classification=_classify_rate(cfr_rate),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                    infraCfr=None,
                    semanticCfr=None,
                ),
                mttr=DeliveryMttrSummary(
                    meanMs=round(mttr_mean) if mttr_mean is not None else None,
                    p50Ms=round(mttr_p50) if mttr_p50 is not None else None,
                    recoveries=len(all_mttr_ms),
                    classification=_classify_duration(mttr_mean),
                    deltaPct=round(rng.uniform(-20, 20), 1),
                ),
                reworkRate=None,
            ),
            series=DeliverySeries(
                deploymentFrequency=freq_points,
                leadTime=lead_points,
                changeFailureRate=cfr_points,
                mttr=mttr_points,
                reworkRate=None,
            ),
        )

    def _apply_metric_filter(
        self, result: DeliveryMetricsResult, metrics: list[str] | None
    ) -> None:
        if not metrics:
            return
        wanted = set(metrics)
        if "deploymentFrequency" not in wanted:
            result["summary"]["deploymentFrequency"] = None
            result["series"]["deploymentFrequency"] = None
        if "leadTime" not in wanted:
            result["summary"]["leadTime"] = None
            result["series"]["leadTime"] = None
        if "changeFailureRate" not in wanted:
            result["summary"]["changeFailureRate"] = None
            result["series"]["changeFailureRate"] = None
        if "mttr" not in wanted:
            result["summary"]["mttr"] = None
            result["series"]["mttr"] = None
        if "reworkRate" not in wanted:
            result["summary"]["reworkRate"] = None
            result["series"]["reworkRate"] = None

    def _build_deployments(
        self,
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        limit: int,
        sort_order: Literal["asc", "desc"],
    ) -> DeliveryDeploymentsResult:
        rng = random.Random(_seed(scope) ^ 0x5EED)
        span = max(1.0, (end - start).total_seconds())
        count = min(limit, rng.randint(8, 40))

        deployments: list[DeliveryDeployment] = []
        for _ in range(count):
            deployed_at = start + timedelta(seconds=rng.random() * span)
            outcome = rng.choice(_OUTCOMES)
            failed = outcome == "failed"
            component = scope["component"] or rng.choice(_MLOPS_TEMPLATES)
            project = scope["project"] or rng.choice(_MLOPS_PROJECTS)
            environment = scope["environment"] or rng.choice(_ENVIRONMENTS)
            lead_time_ms = round(rng.lognormvariate(1.6, 0.7) * 3_600_000)
            change_type = rng.choice(_change_types_for(scope.get("workloadType")))
            is_ml_change = change_type != "infra"
            failure_class: Literal["infra", "semantic"] | None = None
            semantic_type: SemanticFailureType | None = None
            recovery_strategy: RecoveryStrategy | None = None
            if failed:
                failure_class = "semantic" if rng.random() < _SEMANTIC_FAILURE_SHARE else "infra"
                if failure_class == "semantic":
                    semantic_type = rng.choice(_SEMANTIC_TYPES)
                recovery_strategy = rng.choice(_RECOVERY_STRATEGIES)
            lead_time_breakdown = (
                _synthetic_lead_time_breakdown(rng, lead_time_ms) if is_ml_change else None
            )
            eval_bottleneck = (
                max(lead_time_breakdown.items(), key=lambda item: item[1])[0]
                if lead_time_breakdown
                else None
            )

            deployments.append(
                DeliveryDeployment(
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
                    leadTimeMs=lead_time_ms,
                    workloadType=workload_type_for(change_type),
                    changeType=change_type,
                    driftTriggered=is_ml_change and rng.random() < 0.12,
                    evalCoverage=round(rng.uniform(0.5, 1.0), 2) if is_ml_change else None,
                    leadTimeBreakdown=lead_time_breakdown,
                    evalBottleneck=eval_bottleneck,
                    failureClass=failure_class,
                    semanticType=semantic_type,
                    evalScore=round(rng.uniform(0.6, 0.99), 3) if is_ml_change else None,
                    baselineScore=round(rng.uniform(0.6, 0.99), 3) if is_ml_change else None,
                    driftScore=round(rng.uniform(0.0, 0.6), 3) if is_ml_change else None,
                    recoveryStrategy=recovery_strategy,
                    modelVersion=f"v{rng.randint(1, 40)}" if change_type == "model" else None,
                    promptVersion=f"v{rng.randint(1, 40)}" if change_type == "prompt" else None,
                    ragIndexVersion=f"v{rng.randint(1, 40)}"
                    if change_type == "rag_index"
                    else None,
                )
            )

        deployments.sort(key=lambda d: d["deployedAt"], reverse=sort_order == "desc")
        return DeliveryDeploymentsResult(
            deployments=deployments,
            totalCount=len(deployments),
            tookMs=rng.randint(3, 25),
        )
