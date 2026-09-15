"""Tests observability/dora_metrics.py — reads back values from
prometheus_client's default registry rather than mocking it, since these
are simple Histogram/Counter wrappers with no I/O to mock."""

from observability.dora_metrics import (
    COMPLETIONS,
    LEAD_TIME,
    STEP_DURATION,
    record_workflow_completion,
)


def _counter_value(golden_path: str, status: str) -> float:
    return COMPLETIONS.labels(golden_path=golden_path, status=status)._value.get()


def test_record_workflow_completion_increments_success_counter() -> None:
    before = _counter_value("test-gp-success", "success")

    record_workflow_completion(
        golden_path="test-gp-success",
        phase="Succeeded",
        started_at="2026-09-15T01:00:00Z",
        finished_at="2026-09-15T01:05:00Z",
    )

    assert _counter_value("test-gp-success", "success") == before + 1


def test_record_workflow_completion_labels_failed_phase_as_failure() -> None:
    before = _counter_value("test-gp-failure", "failure")

    record_workflow_completion(
        golden_path="test-gp-failure",
        phase="Failed",
        started_at="2026-09-15T01:00:00Z",
        finished_at="2026-09-15T01:05:00Z",
    )

    assert _counter_value("test-gp-failure", "failure") == before + 1


def test_record_workflow_completion_observes_lead_time_seconds() -> None:
    sample_count_before = LEAD_TIME.labels(golden_path="test-gp-lead-time")._sum.get()

    record_workflow_completion(
        golden_path="test-gp-lead-time",
        phase="Succeeded",
        started_at="2026-09-15T01:00:00Z",
        finished_at="2026-09-15T01:05:00Z",
    )

    assert LEAD_TIME.labels(golden_path="test-gp-lead-time")._sum.get() == sample_count_before + 300


def test_record_workflow_completion_skips_lead_time_when_timestamps_missing() -> None:
    sample_count_before = LEAD_TIME.labels(golden_path="test-gp-no-timestamps")._sum.get()

    record_workflow_completion(
        golden_path="test-gp-no-timestamps", phase="Succeeded", started_at=None, finished_at=None
    )

    assert LEAD_TIME.labels(golden_path="test-gp-no-timestamps")._sum.get() == sample_count_before


def test_record_workflow_completion_observes_step_durations() -> None:
    sample_count_before = STEP_DURATION.labels(golden_path="test-gp-steps", step="train")._sum.get()

    record_workflow_completion(
        golden_path="test-gp-steps",
        phase="Succeeded",
        started_at="2026-09-15T01:00:00Z",
        finished_at="2026-09-15T01:05:00Z",
        steps=[
            {
                "name": "train",
                "started_at": "2026-09-15T01:00:00Z",
                "finished_at": "2026-09-15T01:03:00Z",
            }
        ],
    )

    assert (
        STEP_DURATION.labels(golden_path="test-gp-steps", step="train")._sum.get()
        == sample_count_before + 180
    )
