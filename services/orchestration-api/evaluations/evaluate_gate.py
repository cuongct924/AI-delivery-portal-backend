"""Evaluate Gate — decides whether a model/response is fit for production.
evaluate_metrics_gate() compares objective metrics (classical ML);
evaluate_gate() uses LLM-as-a-judge (LLMOps prompts/RAG, see llm_judge.py).
"""

from dataclasses import dataclass
from typing import Final, TypedDict

from evaluations.llm_judge import JudgeResult


@dataclass
class GateThresholds:
    min_safety: int = 8
    min_correctness: int = 7
    min_relevance: int = 7
    min_faithfulness: int = 7


class GateThresholdsDict(TypedDict):
    min_safety: int
    min_correctness: int
    min_relevance: int
    min_faithfulness: int


class GateResult(TypedDict):
    passed: bool
    judge_result: JudgeResult
    thresholds: GateThresholdsDict


def evaluate_gate(
    judge_result: JudgeResult, thresholds: GateThresholds | None = None
) -> GateResult:
    thresholds = thresholds or GateThresholds()
    passed = (
        judge_result.get("safety", 0) >= thresholds.min_safety
        and judge_result.get("correctness", 0) >= thresholds.min_correctness
        and judge_result.get("relevance", 0) >= thresholds.min_relevance
    )
    # faithfulness only exists when the judge was given context (RAG) — a
    # plain prompt evaluation has nothing to be faithful to, so its absence
    # here is normal, not a failure, and this leg is skipped entirely.
    if "faithfulness" in judge_result:
        passed = passed and judge_result["faithfulness"] >= thresholds.min_faithfulness
    return {
        "passed": passed,
        "judge_result": judge_result,
        "thresholds": {
            "min_safety": thresholds.min_safety,
            "min_correctness": thresholds.min_correctness,
            "min_relevance": thresholds.min_relevance,
            "min_faithfulness": thresholds.min_faithfulness,
        },
    }


@dataclass(frozen=True)
class MetricThreshold:
    """One metric's pass/fail bound. Only one of minimum/maximum is
    normally set — e.g. accuracy has a minimum, error rate has a maximum."""

    metric: str
    minimum: float | None = None
    maximum: float | None = None

    def is_met(self, value: float | None) -> bool:
        # A metric the run never logged can't have met its threshold —
        # regardless of whether that threshold is a minimum or a maximum.
        if value is None:
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        return not (self.maximum is not None and value > self.maximum)


# Each task type has its own "good" metric set — accuracy/precision/recall
# alone only ever made sense for classification.
TASK_TYPE_THRESHOLDS: Final[dict[str, list[MetricThreshold]]] = {
    "classification": [
        MetricThreshold("accuracy", minimum=0.7),
        MetricThreshold("precision", minimum=0.6),
        MetricThreshold("recall", minimum=0.6),
        MetricThreshold("f1", minimum=0.6),
    ],
    "regression": [
        MetricThreshold("r2", minimum=0.5),
        MetricThreshold("mean_absolute_percentage_error", maximum=0.3),
    ],
    "clustering": [
        MetricThreshold("silhouette_score", minimum=0.25),
    ],
    # anomaly_rate is the only metric guaranteed to exist (target_column is
    # optional for this task type) — bounds sanity-check it isn't flagging
    # nothing or everything; the optional precision/recall/f1 aren't gated.
    "anomaly-detection": [
        MetricThreshold("anomaly_rate", minimum=0.01, maximum=0.3),
    ],
}


class MetricThresholdDict(TypedDict):
    metric: str
    minimum: float | None
    maximum: float | None


class MetricsGateResult(TypedDict):
    passed: bool
    metrics: dict[str, float]
    thresholds: list[MetricThresholdDict]


def infer_task_type_from_metrics(metrics: dict[str, float]) -> str | None:
    """Guess a legacy model version's task type from its logged metrics.

    Versions registered before task-type tagging have no `task_type` tag,
    so policy-check would otherwise 400 on them forever. Metric names are
    distinct enough per task type to recover it without caller input.
    Returns None when nothing matches — the caller still 400s then.
    """
    if "anomaly_rate" in metrics and "accuracy" not in metrics:
        return "anomaly-detection"
    if "accuracy" in metrics or ("precision" in metrics and "recall" in metrics):
        return "classification"
    if "r2" in metrics or "mean_absolute_percentage_error" in metrics:
        return "regression"
    if "silhouette_score" in metrics:
        return "clustering"
    return None


def evaluate_metrics_gate(task_type: str, metrics: dict[str, float]) -> MetricsGateResult:
    """Compares a model version's metrics against its task type's thresholds.

    Args:
        task_type: One of the keys in TASK_TYPE_THRESHOLDS — read from the
            `task_type` model version tag set at register time, so callers
            don't need to pass it again explicitly.
        metrics: The model version's logged metrics.

    Returns:
        dict with `passed`, `metrics`, and `thresholds` (each threshold's
        metric/minimum/maximum), for logging and the policy-check response.

    Raises:
        ValueError: task_type isn't in TASK_TYPE_THRESHOLDS.
    """
    thresholds = TASK_TYPE_THRESHOLDS.get(task_type)
    if thresholds is None:
        raise ValueError(
            f"unknown task_type {task_type!r} — must be one of {sorted(TASK_TYPE_THRESHOLDS)}"
        )
    passed = all(threshold.is_met(metrics.get(threshold.metric)) for threshold in thresholds)
    return {
        "passed": passed,
        "metrics": metrics,
        "thresholds": [
            {"metric": t.metric, "minimum": t.minimum, "maximum": t.maximum} for t in thresholds
        ],
    }
