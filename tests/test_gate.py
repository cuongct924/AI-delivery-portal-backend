"""services/orchestration-api/evaluations/evaluate_gate.py — pure logic, no external
dependency (dataclasses), runs right away with nothing extra to install."""

import pytest
from evaluations.evaluate_gate import (
    GateThresholds,
    MetricThreshold,
    evaluate_gate,
    evaluate_metrics_gate,
)
from evaluations.llm_judge import JudgeResult


def test_evaluate_gate_passes_when_all_thresholds_met():
    judge_result: JudgeResult = {
        "safety": 9,
        "correctness": 8,
        "relevance": 8,
        "reasoning": "looks good",
    }
    result = evaluate_gate(judge_result)
    assert result["passed"] is True


def test_evaluate_gate_fails_when_safety_below_threshold():
    judge_result: JudgeResult = {
        "safety": 5,
        "correctness": 9,
        "relevance": 9,
        "reasoning": "unsafe",
    }
    result = evaluate_gate(judge_result)
    assert result["passed"] is False


def test_evaluate_gate_respects_custom_thresholds():
    judge_result: JudgeResult = {
        "safety": 6,
        "correctness": 6,
        "relevance": 6,
        "reasoning": "borderline",
    }
    thresholds = GateThresholds(min_safety=5, min_correctness=5, min_relevance=5)
    result = evaluate_gate(judge_result, thresholds)
    assert result["passed"] is True


def test_evaluate_metrics_gate_passes_when_all_thresholds_met():
    metrics = {"accuracy": 0.9, "precision": 0.8, "recall": 0.8, "f1": 0.8}
    result = evaluate_metrics_gate("classification", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_fails_when_accuracy_below_threshold():
    metrics = {"accuracy": 0.4, "precision": 0.9, "recall": 0.9, "f1": 0.9}
    result = evaluate_metrics_gate("classification", metrics)
    assert result["passed"] is False


def test_evaluate_metrics_gate_fails_when_f1_below_threshold():
    metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.4}
    result = evaluate_metrics_gate("classification", metrics)
    assert result["passed"] is False


def test_evaluate_metrics_gate_fails_when_metric_is_missing():
    metrics = {"accuracy": 0.9}
    result = evaluate_metrics_gate("classification", metrics)
    assert result["passed"] is False


def test_evaluate_metrics_gate_uses_regression_thresholds():
    metrics = {"r2": 0.8, "mean_absolute_percentage_error": 0.1}
    result = evaluate_metrics_gate("regression", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_regression_fails_on_high_error():
    metrics = {"r2": 0.8, "mean_absolute_percentage_error": 0.9}
    result = evaluate_metrics_gate("regression", metrics)
    assert result["passed"] is False


def test_evaluate_metrics_gate_uses_clustering_thresholds():
    metrics = {"silhouette_score": 0.5}
    result = evaluate_metrics_gate("clustering", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_uses_ranking_thresholds():
    metrics = {"recall_at_k": 0.25, "ndcg_at_k": 0.35}
    result = evaluate_metrics_gate("ranking", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_ranking_ignores_optional_map_at_k():
    # map_at_k has no threshold entry — logging it shouldn't affect
    # pass/fail either way.
    metrics = {"recall_at_k": 0.25, "ndcg_at_k": 0.35, "map_at_k": 0.01}
    result = evaluate_metrics_gate("ranking", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_uses_anomaly_detection_thresholds():
    metrics = {"anomaly_rate": 0.05}
    result = evaluate_metrics_gate("anomaly-detection", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_anomaly_detection_fails_when_rate_too_high():
    # Flagging almost everything means the model/contamination setting is
    # off, not that it found real anomalies.
    metrics = {"anomaly_rate": 0.9}
    result = evaluate_metrics_gate("anomaly-detection", metrics)
    assert result["passed"] is False


def test_evaluate_metrics_gate_anomaly_detection_ignores_optional_precision():
    # precision/recall/f1 only exist when a label column was provided —
    # not gated on, since anomaly_rate must work standalone too.
    metrics = {"anomaly_rate": 0.05, "precision": 0.1}
    result = evaluate_metrics_gate("anomaly-detection", metrics)
    assert result["passed"] is True


def test_evaluate_metrics_gate_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="unknown task_type"):
        evaluate_metrics_gate("not-a-task-type", {})


def test_metric_threshold_is_met_respects_minimum_and_maximum():
    threshold = MetricThreshold("x", minimum=0.5)
    assert threshold.is_met(0.6) is True
    assert threshold.is_met(0.4) is False
    assert threshold.is_met(None) is False
