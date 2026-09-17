"""Data Drift monitoring — "Setup Model Monitoring" Golden Path. Compares
recent production input data against the model's original training
dataset with Evidently, logs the result to MLflow as a monitoring run tied
to the model, and optionally triggers a retrain. Runs periodically via an
Argo CronWorkflow — a fully separate entrypoint from train.py.

Scope: Data Drift only, not Performance/Error monitoring — those need
production ground-truth labels, which arrive on a separate, slower
feedback loop.
"""

import json
import os
import sys
from pathlib import Path
from typing import Final

import httpx
import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

_DRIFTED_COLUMNS_METRIC_PREFIX: Final = "DriftedColumnsCount"


def compute_drift_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Runs Evidently's DataDriftPreset and returns the share of shared
    columns flagged as drifted (0.0-1.0).

    Args:
        reference: The model's original training data.
        current: Recent production input data to compare against it.

    Raises:
        ValueError: Evidently's report didn't include the expected
            DriftedColumnsCount metric — a version mismatch, not a runtime
            data problem.
    """
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current).dict()
    for metric in result["metrics"]:
        if metric["metric_name"].startswith(_DRIFTED_COLUMNS_METRIC_PREFIX):
            return float(metric["value"]["share"])
    raise ValueError(
        f"Evidently report has no {_DRIFTED_COLUMNS_METRIC_PREFIX} metric — "
        "check the installed evidently version's Report output shape"
    )


def _trigger_retrain(retrain_request_json: str, orchestration_api_url: str) -> None:
    """Calls the same POST /trigger-training the Scaffolder action
    (orchestration:trigger-training) uses — auto-retrain needs no new
    mechanism, just an automated caller instead of a Dev clicking the
    button.

    `retrain_request_json` is the exact JSON body Dev supplied at Setup
    Model Monitoring time — the same request they'd have used to retrigger
    training by hand. Reconstructing it automatically from the
    model's MLflow run metadata would need a different lookup per Golden
    Path/architecture (dataset_uri vs. interactions_uri, tag vs. param for
    task_type, ...) and was cut in favor of this simpler, more reliable
    approach.
    """
    body = json.loads(retrain_request_json)
    response = httpx.post(f"{orchestration_api_url}/trigger-training", json=body, timeout=30.0)
    if response.is_error:
        print(
            f"auto-retrain trigger failed: {response.status_code} {response.text}", file=sys.stderr
        )
        sys.exit(1)


def main() -> None:
    model_name = os.environ["MODEL_NAME"]
    model_version = os.environ["MODEL_VERSION"]
    reference_data_uri = os.environ["REFERENCE_DATA_URI"]
    production_data_uri = os.environ["PRODUCTION_DATA_URI"]
    drift_threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.5"))
    on_drift_detected = os.environ.get("ON_DRIFT_DETECTED", "alert-only")
    retrain_request_json = os.environ.get("RETRAIN_REQUEST_JSON") or None
    orchestration_api_url = os.environ.get(
        "ORCHESTRATION_API_URL", "http://host.docker.internal:8000"
    )
    if on_drift_detected == "auto-retrain" and retrain_request_json is None:
        raise RuntimeError("RETRAIN_REQUEST_JSON is required when ON_DRIFT_DETECTED=auto-retrain")

    reference = pd.read_csv(Path(reference_data_uri.removeprefix("file://")))
    current = pd.read_csv(Path(production_data_uri.removeprefix("file://")))
    # Evidently compares columns present in both — extra id/target columns
    # in either side are harmless, no need to align schemas up front.

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
    )

    with mlflow.start_run(run_name=f"monitor-{model_name}-v{model_version}"):
        mlflow.set_tag("monitoring_model_name", model_name)
        mlflow.set_tag("monitoring_model_version", model_version)
        drift_share = compute_drift_share(reference, current)
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_param("drift_threshold", drift_threshold)
        mlflow.log_param("on_drift_detected", on_drift_detected)

        drifted = drift_share >= drift_threshold
        mlflow.set_tag("drift_detected", str(drifted))
        print(f"drift_share={drift_share:.3f} threshold={drift_threshold} drifted={drifted}")

        if drifted and on_drift_detected == "auto-retrain":
            assert retrain_request_json is not None
            _trigger_retrain(retrain_request_json, orchestration_api_url)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — top-level: any failure must fail the Argo step, not hang.
        print(f"drift monitoring failed: {exc}", file=sys.stderr)
        sys.exit(1)
