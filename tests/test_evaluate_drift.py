"""Tests the delta_pct calculation logic in agents/skills/evaluate_drift.py.

Stubs the "mlflow" package at the sys.modules level before importing —
adapters/ai_platform/mlflow_adapter.py has `import mlflow` at module level, and the real
mlflow package is fairly heavy, which isn't necessary for a pure calculation
unit test like this one.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from agents.skills.evaluate_drift import evaluate_drift  # noqa: E402


def test_evaluate_drift_computes_delta_pct():
    current_metrics = {"accuracy": 0.81, "f1": 0.80}
    baseline_metrics = {"accuracy": 0.90, "f1": 0.80}

    with patch("agents.skills.evaluate_drift.MlflowAdapter") as mock_adapter_cls:
        mock_adapter_cls.return_value.get_model_metrics.side_effect = [
            current_metrics,
            baseline_metrics,
        ]
        result = evaluate_drift("fraud-model", "v2", "v1")

    assert result["accuracy"]["delta_pct"] == pytest.approx(-10.0)
    assert result["f1"]["delta_pct"] == pytest.approx(0.0)


def test_evaluate_drift_skips_metric_missing_in_current():
    with patch("agents.skills.evaluate_drift.MlflowAdapter") as mock_adapter_cls:
        mock_adapter_cls.return_value.get_model_metrics.side_effect = [
            {"accuracy": 0.9},
            {"accuracy": 0.9, "f1": 0.8},
        ]
        result = evaluate_drift("fraud-model", "v2", "v1")

    assert "f1" not in result
    assert "accuracy" in result
