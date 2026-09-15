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


class GateThresholdsDict(TypedDict):
    min_safety: int
    min_correctness: int
    min_relevance: int


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
    return {
        "passed": passed,
        "judge_result": judge_result,
        "thresholds": {
            "min_safety": thresholds.min_safety,
            "min_correctness": thresholds.min_correctness,
            "min_relevance": thresholds.min_relevance,
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
    # anomaly_rate is the only metric guaranteed to exist (train.py's
    # target_column is optional for this task type — see metrics.py) —
    # bounds are a sanity check, not a quality bar: too low means the
    # model isn't flagging anything, too high means it's flagging most of
    # the dataset. precision/recall/f1 get logged too when a label column
    # was provided, but aren't gated on since they're not always present.
    "anomaly-detection": [
        MetricThreshold("anomaly_rate", minimum=0.01, maximum=0.3),
    ],
    # RecSys: warm users/items only, cold-start reported separately.
    "ranking": [
        MetricThreshold("recall_at_k", minimum=0.20),
        MetricThreshold("ndcg_at_k", minimum=0.30),
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
