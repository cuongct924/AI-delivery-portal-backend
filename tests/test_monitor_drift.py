"""infra/argo-workflows/training-image/monitor_drift.py — compute_drift_share()
against real (small, fast) Evidently reports, and _trigger_retrain()'s
httpx call, mocked."""

import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
from monitor_drift import _notify_failure, _trigger_retrain, compute_drift_share, compute_metric


def test_compute_drift_share_is_high_when_distributions_shift() -> None:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)})
    current = pd.DataFrame({"a": rng.normal(6, 1, 200), "b": rng.normal(5, 2, 200)})

    share = compute_drift_share(reference, current)

    # "a" shifted hard, "b" didn't — 1 of 2 columns drifted.
    assert share == pytest.approx(0.5)


def test_compute_drift_share_is_low_when_distributions_match() -> None:
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(5, 2, 300)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(5, 2, 300)})

    share = compute_drift_share(reference, current)

    assert share == pytest.approx(0.0)


@patch("monitor_drift.httpx.post")
def test_trigger_retrain_posts_the_dev_supplied_body(mock_post: MagicMock) -> None:
    mock_post.return_value = MagicMock(is_error=False)
    body = {
        "model_name": "fraud-detection",
        "dataset_uri": "file:///data.csv",
        "task_type": "classification",
    }

    _trigger_retrain(json.dumps(body), "http://orchestration-api.test")

    mock_post.assert_called_once_with(
        "http://orchestration-api.test/trigger-training", json=body, timeout=30.0
    )


@patch("monitor_drift.sys.exit")
@patch("monitor_drift.httpx.post")
def test_trigger_retrain_exits_nonzero_on_http_error(
    mock_post: MagicMock, mock_exit: MagicMock
) -> None:
    mock_post.return_value = MagicMock(is_error=True, status_code=500, text="boom")

    _trigger_retrain("{}", "http://orchestration-api.test")

    mock_exit.assert_called_once_with(1)


def test_compute_metric_supports_classification_and_regression() -> None:
    y_true = pd.Series([1, 0, 1, 1])
    y_pred = [1, 0, 0, 1]

    assert compute_metric("accuracy", y_true, y_pred) == pytest.approx(0.75)
    # weighted f1 across both classes, not the 0.75 accuracy.
    assert compute_metric("f1_score", y_true, y_pred) == pytest.approx(0.7667, abs=1e-3)
    assert compute_metric("rmse", pd.Series([1.0, 2.0]), [1.0, 4.0]) == pytest.approx(2**0.5)


def test_compute_metric_supports_mae_and_r2() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = [1.0, 2.0, 4.0]

    assert compute_metric("mae", y_true, y_pred) == pytest.approx(1 / 3)
    assert compute_metric("r2_score", y_true, y_pred) == pytest.approx(0.5)


def test_compute_metric_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown metric_name"):
        compute_metric("nope", pd.Series([1]), [1])


@patch("monitor_drift.mlflow")
@patch("monitor_drift.pd.read_csv")
def test_run_performance_degradation_uses_per_metric_thresholds(
    mock_read_csv: MagicMock, mock_mlflow: MagicMock
) -> None:
    from monitor_drift import _run_performance_degradation

    # Ground truth is the last column; predictions come from the model.
    mock_read_csv.return_value = pd.DataFrame({"feature": [0, 1, 0, 1], "label": [1, 0, 1, 1]})
    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0, 0, 1]
    mock_mlflow.pyfunc.load_model.return_value = mock_model

    # accuracy=0.75, recall=0.75. A 0.9 threshold on accuracy trips it even
    # though recall's 0.5 threshold would not.
    detected, payload = _run_performance_degradation(
        "m",
        "1",
        pd.DataFrame({"feature": [0, 1, 0, 1]}),
        "file:///labels.csv",
        ["accuracy", "recall"],
        {"accuracy": 0.9, "recall": 0.5},
        0.85,
    )

    assert detected is True
    assert payload["metric_thresholds"] == {"accuracy": 0.9, "recall": 0.5}


@patch("monitor_drift.mlflow")
@patch("monitor_drift.pd.read_csv")
def test_run_performance_degradation_falls_back_to_scalar_threshold(
    mock_read_csv: MagicMock, mock_mlflow: MagicMock
) -> None:
    from monitor_drift import _run_performance_degradation

    mock_read_csv.return_value = pd.DataFrame({"feature": [0, 1, 0, 1], "label": [1, 0, 1, 1]})
    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0, 0, 1]
    mock_mlflow.pyfunc.load_model.return_value = mock_model

    # No explicit threshold for accuracy -> falls back to 0.5, so 0.75 passes.
    detected, payload = _run_performance_degradation(
        "m",
        "1",
        pd.DataFrame({"feature": [0, 1, 0, 1]}),
        "file:///labels.csv",
        ["accuracy"],
        {},
        0.5,
    )

    assert detected is False
    assert payload["metric_thresholds"] == {"accuracy": 0.5}


@patch("monitor_drift.httpx.post")
def test_notify_failure_posts_the_payload(mock_post: MagicMock) -> None:
    mock_post.return_value = MagicMock(is_error=False)

    _notify_failure("http://portal.test/hook", {"model_name": "fraud-detection"})

    mock_post.assert_called_once_with(
        "http://portal.test/hook", json={"model_name": "fraud-detection"}, timeout=30.0
    )


@patch("monitor_drift.httpx.post")
def test_notify_failure_swallows_http_errors(mock_post: MagicMock) -> None:
    mock_post.side_effect = httpx.HTTPError("boom")

    _notify_failure("http://portal.test/hook", {})
