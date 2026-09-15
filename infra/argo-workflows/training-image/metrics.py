"""Metric computation dispatched by task type — each task type has a
different notion of "good" and needs different sklearn.metrics calls."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)


def compute_metrics(
    task_type: str, y_true: ArrayLike | None, y_pred: ArrayLike
) -> dict[str, float]:
    """Computes the metric set for one task type.

    Args:
        task_type: One of "classification", "regression", "clustering",
            "anomaly-detection".
        y_true: Ground-truth labels/targets (classification/regression) or
            the feature matrix used to cluster (clustering — silhouette
            needs the points, not labels). Accepts ndarray/Series/DataFrame —
            callers pass whichever shape their task type produces. `None`
            only for anomaly-detection, whose target column is optional
            (train.py never trains on it either way — see anomaly_rate
            below, which needs no ground truth at all).
        y_pred: Model predictions (classification/regression) or cluster/
            anomaly assignments (clustering/anomaly-detection) — for
            anomaly-detection, already remapped by the caller to 1=anomaly/
            0=normal (sklearn's own IsolationForest/LOF convention is
            -1/1), so it lines up with a labeled column like `is_anomaly`.

    Returns:
        Metric name -> value.

    Raises:
        ValueError: task_type isn't recognized.
    """
    if task_type == "classification":
        # weighted average handles multiclass/imbalance; zero_division stub type is wrong.
        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    if task_type == "regression":
        # r2/MAPE are scale-free, usable as gate thresholds; MAE isn't (logged only).
        return {
            "r2": r2_score(y_true, y_pred),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        }
    if task_type == "clustering":
        # Bounded in [-1, 1] — scale-free, same reasoning as regression above.
        return {"silhouette_score": silhouette_score(y_true, y_pred)}
    if task_type == "anomaly-detection":
        # anomaly_rate needs no ground truth — always computable, so it's
        # what evaluations/gate.py's threshold gates on. precision/recall/f1
        # only get added when a label column was actually provided; IsolationForest/LOF
        # never trained on it either way (see train.py), it's evaluation-only.
        metrics = {"anomaly_rate": float(np.mean(np.asarray(y_pred) == 1))}
        if y_true is not None:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)  # pyright: ignore[reportArgumentType]
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)  # pyright: ignore[reportArgumentType]
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)  # pyright: ignore[reportArgumentType]
        return metrics
    raise ValueError(f"unknown task_type {task_type!r}")
