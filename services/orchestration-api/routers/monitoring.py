"""Model Monitoring API — "Setup Model Monitoring" Golden Path. A separate
router: unlike every other Golden Path, this one doesn't trigger a 1-shot
workflow — it registers a periodic Argo CronWorkflow via
`ArgoAdapter.create_cron_workflow()`.
"""

from typing import Final

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from adapters.factory import get_workflow_adapter

router = APIRouter(tags=["monitoring"])

argo_adapter = get_workflow_adapter()

MONITOR_DRIFT_TEMPLATE: Final[str] = "monitor-drift-golden-path"


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

    argo_adapter.create_cron_workflow(
        cron_workflow_name, request.schedule, MONITOR_DRIFT_TEMPLATE, parameters
    )
    return SetupMonitoringResponse(cron_workflow_name=cron_workflow_name)
