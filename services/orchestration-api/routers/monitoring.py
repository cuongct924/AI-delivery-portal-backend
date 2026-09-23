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
    production_data_uri: str
    schedule: str
    drift_threshold: float = 0.5
    # "alert-only" | "auto-retrain" — Dev-facing on purpose, auto-retrain
    # has real risk if the drift check false-positives.
    on_drift_detected: str = "alert-only"
    # Required when on_drift_detected="auto-retrain" — the exact JSON body
    # Dev would have POSTed to /trigger-training by hand.
    retrain_request_json: str | None = None


class SetupMonitoringResponse(BaseModel):
    cron_workflow_name: str


@router.post("/setup-monitoring", response_model=SetupMonitoringResponse)
def setup_monitoring(
    request: SetupMonitoringRequest, user: dict = Depends(get_current_user)
) -> SetupMonitoringResponse:
    if request.on_drift_detected == "auto-retrain" and request.retrain_request_json is None:
        raise ValueError("retrain_request_json is required when on_drift_detected='auto-retrain'")

    # Deterministic name — re-running Setup for the same model updates the
    # existing schedule/threshold instead of creating a duplicate CronWorkflow.
    cron_workflow_name = f"monitor-{request.model_name}"
    parameters = {
        "model-name": request.model_name,
        "model-version": request.model_version,
        "reference-data-uri": request.reference_data_uri,
        "production-data-uri": request.production_data_uri,
        "drift-threshold": str(request.drift_threshold),
        "on-drift-detected": request.on_drift_detected,
    }
    if request.retrain_request_json is not None:
        parameters["retrain-request-json"] = request.retrain_request_json

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
