"""Real IDeliveryObserverAdapter — queries OpenChoreo's own in-cluster
Prometheus (already scraping orchestration-api's `/metrics`, which
`services/orchestration-api/observability/dora_metrics.py` already populates
from real gate checks, deploys/rollbacks, and drift-driven recoveries in
routers/models.py|rag.py|prompts.py) as the primary source for the 4 DORA
tiles and the ML/LLM split, and MLflow as a secondary source for continuous
values Prometheus counters can't carry (drift score, model version).

Prometheus's counters/histograms have no per-event detail (commit, incident
id, ...), so they can't answer `query_deployments`' itemized rows; that
still reads the same captured Golden Path run snapshot
MockDeliveryObserverAdapter's `captured` mode uses, enriched here with a
live MLflow drift lookup per row.
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

import httpx
import pandas as pd

from adapters.ai_platform.mlflow_adapter import MlflowAdapter
from adapters.ai_platform.mock_model_registry_adapter import MockModelRegistryAdapter
from adapters.delivery._captured_runs import build_deployment_rows, load_captured_runs
from adapters.delivery.interfaces import (
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
    DeliveryScope,
    DeliverySeries,
    DeliverySummary,
    DeliveryWindow,
    DoraGranularity,
    IDeliveryObserverAdapter,
)

_DAYS_PER_BUCKET: Final[dict[str, int]] = {"daily": 1, "weekly": 7, "monthly": 30}


def _bucket_starts(start: datetime, end: datetime, granularity: str) -> list[datetime]:
    step = timedelta(days=_DAYS_PER_BUCKET[granularity])
    buckets: list[datetime] = []
    cursor = start
    while cursor < end:
        buckets.append(cursor)
        cursor += step
    return buckets or [start]


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


def _quantile_query(bucket_metric: str, quantile: float) -> str:
    return f"histogram_quantile({quantile}, sum(rate({bucket_metric}[__STEP__s])) by (le))"


class PrometheusDeliveryObserverAdapter(IDeliveryObserverAdapter):
    def __init__(
        self,
        model_registry_adapter: MlflowAdapter | MockModelRegistryAdapter,
        prometheus_url: str | None = None,
    ) -> None:
        # Captured runs are always changeType="model" today (see
        # _captured_runs.py) — no code path yet needs a prompt/rag-index
        # version registry, so this doesn't take one; add it if/when the
        # capture script starts producing LLMOps rows.
        self._model_registry_adapter = model_registry_adapter
        self._prometheus_url = prometheus_url or os.getenv(
            "PROMETHEUS_URL", "http://localhost:9090"
        )

    def query_metrics(
        self,
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        granularity: DoraGranularity,
        metrics: list[str] | None,
    ) -> DeliveryMetricsResult:
        # dora_metrics.py's counters carry track/subject_type/subject_id, not
        # namespace/project/component/environment — scope filtering isn't
        # possible against today's label set, so this returns portal-wide
        # figures regardless of `scope` (echoed back for contract shape only).
        window_days = max(1, (end - start).days)
        buckets = _bucket_starts(start, end, granularity)
        step_seconds = _DAYS_PER_BUCKET[granularity] * 86_400

        completed = self._range_by_bucket(
            'sum(increase(golden_path_completions_total{status="success"}[__STEP__s]))',
            buckets,
            end,
            step_seconds,
        )
        failed = self._range_by_bucket(
            'sum(increase(golden_path_completions_total{status="failure"}[__STEP__s]))',
            buckets,
            end,
            step_seconds,
        )
        semantic_failed = self._range_by_bucket(
            'sum(increase(dora_gate_evaluations_total{passed="false"}[__STEP__s]))',
            buckets,
            end,
            step_seconds,
        )
        lead_p50 = self._range_by_bucket(
            _quantile_query("golden_path_lead_time_seconds_bucket", 0.5), buckets, end, step_seconds
        )
        lead_p75 = self._range_by_bucket(
            _quantile_query("golden_path_lead_time_seconds_bucket", 0.75),
            buckets,
            end,
            step_seconds,
        )
        lead_p95 = self._range_by_bucket(
            _quantile_query("golden_path_lead_time_seconds_bucket", 0.95),
            buckets,
            end,
            step_seconds,
        )
        mttr_p50 = self._range_by_bucket(
            _quantile_query("dora_incident_recovery_seconds_bucket", 0.5),
            buckets,
            end,
            step_seconds,
        )
        mttr_mean = self._range_by_bucket(
            "sum(rate(dora_incident_recovery_seconds_sum[__STEP__s])) / "
            "sum(rate(dora_incident_recovery_seconds_count[__STEP__s]))",
            buckets,
            end,
            step_seconds,
        )
        mttr_count = self._range_by_bucket(
            "sum(increase(dora_incident_recovery_seconds_count[__STEP__s]))",
            buckets,
            end,
            step_seconds,
        )

        freq_points = [
            DeliveryFrequencyPoint(bucketStart=b.isoformat(), count=round(completed[i]))
            for i, b in enumerate(buckets)
        ]
        cfr_points = [
            DeliveryChangeFailureRatePoint(
                bucketStart=b.isoformat(),
                rate=(failed[i] / (completed[i] + failed[i]))
                if (completed[i] + failed[i])
                else 0.0,
                failed=round(failed[i]),
                total=round(completed[i] + failed[i]),
            )
            for i, b in enumerate(buckets)
        ]
        lead_points = [
            DeliveryLeadTimePoint(
                bucketStart=b.isoformat(),
                p50Ms=round(lead_p50[i] * 1000),
                p75Ms=round(lead_p75[i] * 1000),
                p95Ms=round(lead_p95[i] * 1000),
            )
            for i, b in enumerate(buckets)
            if lead_p50[i] > 0
        ]
        mttr_points = [
            DeliveryMttrPoint(
                bucketStart=b.isoformat(),
                meanMs=round(mttr_mean[i] * 1000),
                p50Ms=round(mttr_p50[i] * 1000),
                count=round(mttr_count[i]),
            )
            for i, b in enumerate(buckets)
            if mttr_count[i] > 0
        ]

        total_deployments = round(sum(completed))
        total_failed = round(sum(failed))
        total_semantic_failed = round(sum(semantic_failed))
        total_changes = total_deployments + total_failed
        per_day = total_deployments / window_days
        cfr_rate = (total_failed / total_changes) if total_changes else 0.0
        all_lead_p50 = [v for v in lead_p50 if v > 0]
        all_mttr_mean = [v for v in mttr_mean if v > 0]

        result = DeliveryMetricsResult(
            dataAvailability=DeliveryAvailability(
                collecting=True,
                deliveryEvents=total_changes > 0,
                evalPipeline=any(v > 0 for v in semantic_failed) or total_semantic_failed > 0,
                driftMonitor=any(v > 0 for v in mttr_count),
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
                    deltaPct=None,
                ),
                leadTime=DeliveryLeadTimeSummary(
                    p50Ms=round(sum(all_lead_p50) / len(all_lead_p50) * 1000)
                    if all_lead_p50
                    else None,
                    p95Ms=round(max(lead_p95) * 1000) if any(v > 0 for v in lead_p95) else None,
                    coverage=1.0 if total_changes else 0.0,
                    classification=_classify_duration(
                        (sum(all_lead_p50) / len(all_lead_p50) * 1000) if all_lead_p50 else None
                    ),
                    deltaPct=None,
                ),
                changeFailureRate=DeliveryChangeFailureRateSummary(
                    rate=round(cfr_rate, 4),
                    failed=total_failed,
                    total=total_changes,
                    classification=_classify_rate(cfr_rate),
                    deltaPct=None,
                    infraCfr=round((total_failed - total_semantic_failed) / total_changes, 4)
                    if total_changes
                    else None,
                    semanticCfr=round(total_semantic_failed / total_changes, 4)
                    if total_changes
                    else None,
                ),
                mttr=DeliveryMttrSummary(
                    meanMs=round(sum(all_mttr_mean) / len(all_mttr_mean) * 1000)
                    if all_mttr_mean
                    else None,
                    p50Ms=round(
                        sum(v for v in mttr_p50 if v > 0)
                        / max(1, sum(1 for v in mttr_p50 if v > 0))
                        * 1000
                    )
                    if any(v > 0 for v in mttr_p50)
                    else None,
                    recoveries=round(sum(mttr_count)),
                    classification=_classify_duration(
                        (sum(all_mttr_mean) / len(all_mttr_mean) * 1000) if all_mttr_mean else None
                    ),
                    deltaPct=None,
                ),
            ),
            series=DeliverySeries(
                deploymentFrequency=freq_points,
                leadTime=lead_points,
                changeFailureRate=cfr_points,
                mttr=mttr_points,
            ),
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
        runs = load_captured_runs()
        rows = build_deployment_rows(runs, scope, start, end, limit, sort_order)
        enriched = [self._enrich_with_drift(row) for row in rows]
        return DeliveryDeploymentsResult(deployments=enriched, totalCount=len(enriched), tookMs=0)

    def _enrich_with_drift(self, row: DeliveryDeployment) -> DeliveryDeployment:
        """Looks up the most recent drift-monitoring MLflow run for this
        row's model (keyed by projectName — the closest proxy the captured
        run data has to a registered model name) at or before its
        deployedAt, per the tags infra/argo-workflows/training-image/
        monitor_drift.py:97-107 writes. modelVersion/evalScore stay None —
        no interface method resolves a point-in-time model version from a
        captured run name (see plan's known-gaps section)."""
        model_name = row["projectName"]
        deployed_at = row["deployedAt"]
        runs_df = self._model_registry_adapter.search_runs(
            filter_string=f"tags.monitoring_model_name = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=5,
        )
        if runs_df.empty:
            return row
        for _, run in runs_df.iterrows():
            start_time = run.get("start_time")
            if not isinstance(start_time, pd.Timestamp):
                continue
            if start_time.isoformat() > deployed_at:
                continue
            drift_share = run.get("metrics.drift_share")
            row["driftScore"] = float(drift_share) if isinstance(drift_share, int | float) else None
            row["driftTriggered"] = str(run.get("tags.drift_detected")) == "True"
            row["recoveryStrategy"] = (
                "retrain" if run.get("params.on_drift_detected") == "auto-retrain" else None
            )
            break
        return row

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

    def _range_by_bucket(
        self, promql_template: str, buckets: list[datetime], end: datetime, step_seconds: int
    ) -> list[float]:
        """Runs `promql_template` (with `__STEP__` substituted for the bucket
        width) as a query_range and returns one value per bucket, aligned by
        index rather than exact timestamp — acceptable since each returned
        sample already represents one `step_seconds`-wide window."""
        promql = promql_template.replace("__STEP__", str(step_seconds))
        window_start = buckets[0]
        response = httpx.get(
            f"{self._prometheus_url}/api/v1/query_range",
            params={
                "query": promql,
                "start": window_start.timestamp(),
                "end": end.timestamp(),
                "step": f"{step_seconds}s",
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json().get("data", {}).get("result", [])
        values = [float(v) for _, v in result[0]["values"]] if result else []
        padded = values + [0.0] * (len(buckets) - len(values))
        return padded[: len(buckets)]
