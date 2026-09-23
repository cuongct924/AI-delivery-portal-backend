"""Model monitoring — "Setup Model Monitoring" Golden Path. Two signals,
selected by MONITORING_TYPE:

- "data-drift" (default): compares recent production input data against the
  model's original training dataset with Evidently.
- "performance-degradation": scores recent production data with the model
  and compares the chosen metric against delayed ground-truth labels.

Either way the result is logged to MLflow as a monitoring run tied to the
model, an optional failure webhook is POSTed, and an optional retrain is
triggered. Runs periodically via an Argo CronWorkflow — a fully separate
entrypoint from train.py.
"""

import json
import os
import sys
from pathlib import Path
from typing import Final

import httpx
import mlflow
import mlflow.pyfunc
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

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


def compute_metric(metric_name: str, y_true: pd.Series, y_pred: object) -> float:
    """Computes one named performance metric.

    Classification metrics use a weighted average (multiclass/imbalance);
    "rmse" is regression-only.

    Raises:
        ValueError: metric_name isn't recognized.
    """
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric_name == "precision":
        # zero_division stub type is wrong, same as metrics.py.
        return float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)  # pyright: ignore[reportArgumentType]
        )
    if metric_name == "recall":
        return float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)  # pyright: ignore[reportArgumentType]
        )
    if metric_name == "f1_score":
        return float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)  # pyright: ignore[reportArgumentType]
        )
    if metric_name == "rmse":
        return float(root_mean_squared_error(y_true, y_pred))
    raise ValueError(f"unknown metric_name {metric_name!r}")


def _notify_failure(webhook_url: str, payload: dict[str, object]) -> None:
    """POSTs a detection summary to the Portal's monitoring webhook. A
    webhook failure is logged, not fatal — the MLflow run already recorded
    the result."""
    try:
        response = httpx.post(webhook_url, json=payload, timeout=30.0)
        if response.is_error:
            print(
                f"failure webhook returned {response.status_code}: {response.text}", file=sys.stderr
            )
    except httpx.HTTPError as exc:
        print(f"failure webhook unreachable: {exc}", file=sys.stderr)


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


def _run_data_drift(
    reference: pd.DataFrame, current: pd.DataFrame, drift_threshold: float
) -> tuple[bool, dict[str, object]]:
    """Logs a drift run and returns (detected, webhook payload)."""
    drift_share = compute_drift_share(reference, current)
    mlflow.log_metric("drift_share", drift_share)
    mlflow.log_param("drift_threshold", drift_threshold)
    drifted = drift_share >= drift_threshold
    mlflow.set_tag("drift_detected", str(drifted))
    print(f"drift_share={drift_share:.3f} threshold={drift_threshold} drifted={drifted}")
    return drifted, {"drift_share": drift_share, "drift_threshold": drift_threshold}


def _run_performance_degradation(
    model_name: str,
    model_version: str,
    current: pd.DataFrame,
    ground_truth_data_uri: str,
    metric_name: str,
    min_metric_threshold: float,
) -> tuple[bool, dict[str, object]]:
    """Scores production data with the model, compares the metric against
    delayed ground-truth labels (last column of the ground-truth CSV,
    aligned by row position), and returns (detected, webhook payload)."""
    ground_truth = pd.read_csv(Path(ground_truth_data_uri.removeprefix("file://")))
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
    predictions = model.predict(current)
    y_true = ground_truth.iloc[:, -1]
    metric_value = compute_metric(metric_name, y_true, predictions)
    mlflow.log_metric(metric_name, metric_value)
    mlflow.log_param("min_metric_threshold", min_metric_threshold)
    degraded = metric_value < min_metric_threshold
    mlflow.set_tag("degraded", str(degraded))
    print(f"{metric_name}={metric_value:.3f} min={min_metric_threshold} degraded={degraded}")
    return degraded, {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "min_metric_threshold": min_metric_threshold,
    }


def main() -> None:
    model_name = os.environ["MODEL_NAME"]
    model_version = os.environ["MODEL_VERSION"]
    reference_data_uri = os.environ["REFERENCE_DATA_URI"]
    production_data_uri = os.environ["PRODUCTION_DATA_URI"]
    monitoring_type = os.environ.get("MONITORING_TYPE", "data-drift")
    drift_threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.5"))
    ground_truth_data_uri = os.environ.get("GROUND_TRUTH_DATA_URI") or None
    metric_name = os.environ.get("METRIC_NAME") or None
    min_metric_threshold = float(os.environ.get("MIN_METRIC_THRESHOLD", "0.85"))
    on_drift_detected = os.environ.get("ON_DRIFT_DETECTED", "alert-only")
    retrain_request_json = os.environ.get("RETRAIN_REQUEST_JSON") or None
    failure_webhook_url = os.environ.get("FAILURE_WEBHOOK_URL") or None
    orchestration_api_url = os.environ.get(
        "ORCHESTRATION_API_URL", "http://host.docker.internal:8000"
    )
    if on_drift_detected == "auto-retrain" and retrain_request_json is None:
        raise RuntimeError("RETRAIN_REQUEST_JSON is required when ON_DRIFT_DETECTED=auto-retrain")
    if monitoring_type == "performance-degradation" and (
        ground_truth_data_uri is None or metric_name is None
    ):
        raise RuntimeError(
            "GROUND_TRUTH_DATA_URI and METRIC_NAME are required "
            "when MONITORING_TYPE=performance-degradation"
        )

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
        mlflow.set_tag("monitoring_type", monitoring_type)
        mlflow.log_param("on_drift_detected", on_drift_detected)

        if monitoring_type == "performance-degradation":
            assert ground_truth_data_uri is not None and metric_name is not None
            detected, payload = _run_performance_degradation(
                model_name,
                model_version,
                current,
                ground_truth_data_uri,
                metric_name,
                min_metric_threshold,
            )
        else:
            detected, payload = _run_data_drift(reference, current, drift_threshold)

        if detected:
            if failure_webhook_url is not None:
                _notify_failure(
                    failure_webhook_url,
                    {
                        "model_name": model_name,
                        "model_version": model_version,
                        "monitoring_type": monitoring_type,
                        **payload,
                    },
                )
            if on_drift_detected == "auto-retrain":
                assert retrain_request_json is not None
                _trigger_retrain(retrain_request_json, orchestration_api_url)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — top-level: any failure must fail the Argo step, not hang.
        print(f"drift monitoring failed: {exc}", file=sys.stderr)
        sys.exit(1)
