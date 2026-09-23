"""Model Monitoring API — "Setup Model Monitoring" Golden Path. A separate
router: unlike every other Golden Path, this one doesn't trigger a 1-shot
workflow — it registers a periodic cron/WorkflowRun via
`IWorkflowAdapter.create_cron_workflow()`.

Known gap: the OpenChoreo workflow backend that replaced Argo Server (see
docs/openchoreo-workflow-migration-plan.md) has no `CronWorkflow`
equivalent — `OpenChoreoWorkflowAdapter.create_cron_workflow()` raises
`NotImplementedError` until scheduled monitoring gets its own OpenChoreo
scheduled-task design. Mocking the workflow adapter is the only way this
endpoint works today.
"""

from typing import Final

from auth.thunder import get_current_user
from costs.events import record_cost_event
from costs.pricing import CPU_HOUR_USD, MONITOR_RUN_HOURS
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from adapters.factory import get_workflow_adapter

router = APIRouter(tags=["monitoring"])

workflow_adapter = get_workflow_adapter()

MONITOR_DRIFT_TEMPLATE: Final[str] = "monitor-drift-golden-path"

# Where the platform's managed prediction log lands for a deployed model —
# the production side used when Dev picks the "managed-prediction-log" source.
_MANAGED_PREDICTION_LOG_URI: Final[str] = "file:///mnt/data/{model_name}/prediction-log.csv"

# Where the platform's managed delayed-label stream lands — the ground-truth
# side used when Dev picks the "managed-label-log" source.
_MANAGED_LABEL_LOG_URI: Final[str] = "file:///mnt/data/{model_name}/label-log.csv"

# Cron preset → runs per month, for pricing the recurring monitoring job.
_RUNS_PER_MONTH: Final[dict[str, int]] = {
    "0 * * * *": 720,  # hourly
    "0 0 * * *": 30,  # daily
    "0 0 * * 0": 4,  # weekly
}


class SetupMonitoringRequest(BaseModel):
    model_name: str
    model_version: str
    reference_data_uri: str
    # "managed-prediction-log" (default) | "custom-uri" — mirrors the Portal
    # template. Managed resolves to the platform's own prediction log path.
    production_data_source: str = "managed-prediction-log"
    # Only required when production_data_source="custom-uri".
    production_data_uri: str | None = None
    schedule: str
    # "data-drift" | "performance-degradation" — the latter needs delayed
    # ground-truth labels and a metric to watch.
    monitoring_type: str = "data-drift"
    drift_threshold: float = 0.5
    # "managed-label-log" (default) | "custom-uri" — only used when
    # monitoring_type="performance-degradation".
    ground_truth_data_source: str = "managed-label-log"
    # Only required when ground_truth_data_source="custom-uri".
    ground_truth_data_uri: str | None = None
    # Required when monitoring_type="performance-degradation".
    metric_name: str | None = None
    min_metric_threshold: float = 0.85
    # "alert-only" | "auto-retrain" — Dev-facing on purpose, auto-retrain
    # has real risk if the drift check false-positives.
    on_drift_detected: str = "alert-only"
    # Required when on_drift_detected="auto-retrain" — the exact JSON body
    # Dev would have POSTed to /trigger-training by hand.
    retrain_request_json: str | None = None
    # Portal endpoint the monitoring run POSTs to when it detects drift/
    # degradation — optional, so a run without it just logs to MLflow.
    failure_webhook_url: str | None = None


def _resolve_production_data_uri(request: SetupMonitoringRequest) -> str:
    """Resolves the production data URI from the chosen source.

    An explicit URI always wins; otherwise the managed prediction log path
    is derived from the model name.

    Raises:
        ValueError: production_data_source="custom-uri" but no URI given.
    """
    if request.production_data_uri:
        return request.production_data_uri
    if request.production_data_source == "managed-prediction-log":
        return _MANAGED_PREDICTION_LOG_URI.format(model_name=request.model_name)
    raise ValueError("production_data_uri is required when production_data_source='custom-uri'")


def _resolve_ground_truth_data_uri(request: SetupMonitoringRequest) -> str | None:
    """Resolves the ground-truth URI for performance-degradation monitoring.

    Returns None for data-drift (no labels needed). An explicit URI always
    wins; otherwise the managed label log path is derived from the model name.

    Raises:
        ValueError: performance-degradation with ground_truth_data_source=
            "custom-uri" but no URI given.
    """
    if request.monitoring_type != "performance-degradation":
        return None
    if request.ground_truth_data_uri:
        return request.ground_truth_data_uri
    if request.ground_truth_data_source == "managed-label-log":
        return _MANAGED_LABEL_LOG_URI.format(model_name=request.model_name)
    raise ValueError("ground_truth_data_uri is required when ground_truth_data_source='custom-uri'")


class SetupMonitoringResponse(BaseModel):
    cron_workflow_name: str


@router.post("/setup-monitoring", response_model=SetupMonitoringResponse)
def setup_monitoring(
    request: SetupMonitoringRequest, user: dict = Depends(get_current_user)
) -> SetupMonitoringResponse:
    if request.on_drift_detected == "auto-retrain" and request.retrain_request_json is None:
        raise ValueError("retrain_request_json is required when on_drift_detected='auto-retrain'")
    if request.monitoring_type == "performance-degradation" and request.metric_name is None:
        raise ValueError("metric_name is required when monitoring_type='performance-degradation'")

    # Deterministic name — re-running Setup for the same model updates the
    # existing schedule/threshold instead of creating a duplicate CronWorkflow.
    cron_workflow_name = f"monitor-{request.model_name}"
    parameters = {
        "model-name": request.model_name,
        "model-version": request.model_version,
        "reference-data-uri": request.reference_data_uri,
        "production-data-uri": _resolve_production_data_uri(request),
        "monitoring-type": request.monitoring_type,
        "drift-threshold": str(request.drift_threshold),
        "min-metric-threshold": str(request.min_metric_threshold),
        "on-drift-detected": request.on_drift_detected,
    }
    ground_truth_data_uri = _resolve_ground_truth_data_uri(request)
    if ground_truth_data_uri is not None:
        parameters["ground-truth-data-uri"] = ground_truth_data_uri
    if request.metric_name is not None:
        parameters["metric-name"] = request.metric_name
    if request.retrain_request_json is not None:
        parameters["retrain-request-json"] = request.retrain_request_json
    if request.failure_webhook_url is not None:
        parameters["failure-webhook-url"] = request.failure_webhook_url

    workflow_adapter.create_cron_workflow(
        cron_workflow_name, request.schedule, MONITOR_DRIFT_TEMPLATE, parameters
    )

    # Attribute the recurring monitoring compute to the model's run stage,
    # priced for a month of runs at the schedule's cadence.
    runs = _RUNS_PER_MONTH.get(request.schedule, 30)
    hours = runs * MONITOR_RUN_HOURS
    record_cost_event(
        stage="run",
        artifact_kind="model",
        artifact_id=request.model_name,
        version=request.model_version,
        environment="production",
        cost_usd=hours * CPU_HOUR_USD,
        quantity=float(runs),
        unit="job-run",
        unit_price=MONITOR_RUN_HOURS * CPU_HOUR_USD,
        source="workflow-estimate",
        run_id=cron_workflow_name,
    )

    return SetupMonitoringResponse(cron_workflow_name=cron_workflow_name)
