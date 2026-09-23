"""Captured Golden Path runs — shared by MockDeliveryObserverAdapter (its
`captured` data source) and PrometheusDeliveryObserverAdapter (its base
deployment-row source, since Prometheus counters have no itemized per-event
detail to reconstruct a breakdown table row from). Produced by
scripts/capture-delivery-insights-runs.sh.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypedDict

from adapters.delivery.interfaces import DeliveryDeployment, DeliveryScope, workload_type_for

REAL_RUNS_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent
    / "services"
    / "orchestration-api"
    / "mock_data"
    / "delivery_insights_runs.json"
)

# Capture script only pulls MLOps runs — rows are always model changes for now.
_CAPTURED_CHANGE_TYPE: Final = "model"


class CapturedStep(TypedDict):
    name: str
    started_at: str
    finished_at: str


class CapturedRun(TypedDict):
    name: str
    started_at: str
    finished_at: str
    outcome: str
    steps: list[CapturedStep]


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def load_captured_runs(path: Path = REAL_RUNS_PATH) -> list[CapturedRun]:
    """Loads the captured Golden Path runs, or [] when the file is absent."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    runs: list[CapturedRun] = raw.get("runs", [])
    return [r for r in runs if r.get("started_at") and r.get("finished_at")]


def build_deployment_rows(
    runs: list[CapturedRun],
    scope: DeliveryScope,
    start: datetime,
    end: datetime,
    limit: int,
    sort_order: Literal["asc", "desc"],
) -> list[DeliveryDeployment]:
    """Maps captured runs in [start, end) to deployment rows. ML/LLM fields
    default to None/False here — callers that have a real enrichment source
    (MLflow tags, version registries) fill them in afterwards."""
    in_window = [r for r in runs if start <= parse_iso(r["started_at"]) < end]
    in_window.sort(key=lambda r: r["started_at"], reverse=sort_order == "desc")

    rows: list[DeliveryDeployment] = []
    for run in in_window[:limit]:
        started = parse_iso(run["started_at"])
        finished = parse_iso(run["finished_at"])
        failed = run["outcome"] != "success"
        rows.append(
            DeliveryDeployment(
                deployedAt=finished.isoformat(),
                projectName=scope["project"] or "telco-fraud-detection",
                componentName=scope["component"] or "train-track-register",
                environmentName=scope["environment"] or "development",
                componentRelease=run["name"],
                commit="",
                outcome="failed" if failed else "success",
                failedBy="train-step" if failed else "",
                failureReason="training failed" if failed else "",
                incidentId=f"INC-{run['name'][-6:]}" if failed else "",
                leadTimeMs=round((finished - started).total_seconds() * 1000),
                workloadType=workload_type_for(_CAPTURED_CHANGE_TYPE),
                changeType=_CAPTURED_CHANGE_TYPE,
                driftTriggered=False,
                evalCoverage=None,
                leadTimeBreakdown=None,
                evalBottleneck=None,
                failureClass=None,
                semanticType=None,
                evalScore=None,
                baselineScore=None,
                driftScore=None,
                recoveryStrategy=None,
                modelVersion=None,
                promptVersion=None,
                ragIndexVersion=None,
            )
        )
    return rows
