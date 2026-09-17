"""infra/argo-workflows/training-image/metrics.py — compute_metrics dispatch."""

import numpy as np
import pytest
from metrics import compute_metrics


def test_classification_metrics_shape() -> None:
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    result = compute_metrics("classification", y_true, y_pred)
    assert set(result) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0


def test_regression_metrics_shape() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 305.0])
    result = compute_metrics("regression", y_true, y_pred)
    assert set(result) == {"r2", "mean_absolute_percentage_error", "mean_absolute_error"}
    assert result["mean_absolute_error"] == pytest.approx(25 / 3)


def test_clustering_metrics_shape() -> None:
    x = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    labels = np.array([0, 0, 1, 1])
    result = compute_metrics("clustering", x, labels)
    assert set(result) == {"silhouette_score"}
    assert -1.0 <= result["silhouette_score"] <= 1.0


def test_anomaly_detection_metrics_without_labels() -> None:
    # No target_column set — anomaly_rate is the only metric that needs no
    # ground truth (see evaluate_gate.py's threshold, which only gates on this one).
    y_pred = np.array([0, 0, 1, 0, 0])
    result = compute_metrics("anomaly-detection", None, y_pred)
    assert set(result) == {"anomaly_rate"}
    assert result["anomaly_rate"] == pytest.approx(0.2)


def test_anomaly_detection_metrics_with_labels() -> None:
    y_true = np.array([0, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0])
    result = compute_metrics("anomaly-detection", y_true, y_pred)
    assert set(result) == {"anomaly_rate", "precision", "recall", "f1"}
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)


def test_unknown_task_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown task_type"):
        compute_metrics("not-a-task-type", np.array([1]), np.array([1]))
