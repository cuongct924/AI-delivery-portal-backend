"""RQ1 (Velocity & Lead Time) Prometheus metrics — derived from Argo
Workflow timestamps `IWorkflowAdapter.get_workflow_status()` already
fetches, not scraped separately from Argo's own controller.

`infra/monitoring/prometheus.yml` notes that docker-compose's Prometheus
and the k3d cluster running Argo sit on separate Docker networks with no
bridge, so scraping Argo's controller metrics endpoint directly isn't
reachable. Computing duration here and exposing it via orchestration-api's
own `/metrics` (already scraped, see main.py's Instrumentator) avoids
needing that bridge at all.

Callers must dedupe repeated status polls themselves — `record_workflow_completion`
increments/observes once per call, so anything polling a workflow's status
in a loop (as the frontend does) must only call this once the workflow
reaches a terminal phase.

DORA Metrics for both MLOps and LLMOps tracks — unified metric names
with `track` label ("mlops" | "llmops") so a single dashboard variable
can filter across both tracks on the same panel.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import TypedDict

from prometheus_client import Counter, Gauge, Histogram

STEP_DURATION = Histogram(
    "golden_path_step_duration_seconds",
    "Duration of one Golden Path workflow step, from Argo node timestamps.",
    labelnames=["golden_path", "step"],
)

LEAD_TIME = Histogram(
    "golden_path_lead_time_seconds",
    "Wall-clock time from workflow trigger to terminal status, per Golden Path.",
    labelnames=["golden_path"],
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600, 7200),
)

COMPLETIONS = Counter(
    "golden_path_completions_total",
    "Golden Path workflow completions, by outcome.",
    labelnames=["golden_path", "status"],
)

GATE_EVALUATIONS = Counter(
    "dora_gate_evaluations_total",
    "Evaluate Gate outcomes, by track and subject.",
    labelnames=["track", "subject_type", "subject_id", "passed"],
)

DEPLOYMENT_EVENTS = Counter(
    "dora_deployment_events_total",
    "Deploy/rollback events, by track and subject.",
    labelnames=["track", "subject_type", "subject_id", "event_type"],
)

INCIDENT_RECOVERY = Histogram(
    "dora_incident_recovery_seconds",
    "Time from a detected failure signal to the next remediation event, by track.",
    labelnames=["track", "subject_type", "subject_id"],
    buckets=(60, 300, 600, 1800, 3600, 7200, 21600, 86400),
)

LLM_SPEND_USD = Gauge(
    "dora_llm_spend_usd",
    "LLM API spend in USD, by model.",
    labelnames=["model"],
)


class StepTimingInput(TypedDict):
    name: str
    started_at: str | None
    finished_at: str | None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def record_workflow_completion(
    golden_path: str,
    phase: str | None,
    started_at: str | None,
    finished_at: str | None,
    steps: Sequence[StepTimingInput] | None = None,
) -> None:
    """Records 1 completion + (when timestamps are present) 1 lead-time
    observation + 1 step-duration observation per step. Call exactly once
    per terminal workflow — the caller (routers/models.py) owns dedup."""
    status = "success" if phase == "Succeeded" else "failure"
    COMPLETIONS.labels(golden_path=golden_path, status=status).inc()

    start = _parse_timestamp(started_at)
    finish = _parse_timestamp(finished_at)
    if start is not None and finish is not None:
        LEAD_TIME.labels(golden_path=golden_path).observe((finish - start).total_seconds())

    for step in steps or []:
        step_start = _parse_timestamp(step.get("started_at"))
        step_finish = _parse_timestamp(step.get("finished_at"))
        if step_start is not None and step_finish is not None:
            STEP_DURATION.labels(golden_path=golden_path, step=step["name"]).observe(
                (step_finish - step_start).total_seconds()
            )
