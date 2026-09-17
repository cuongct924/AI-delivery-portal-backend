"""Tests observability/dora_metrics.py — reads back values from
prometheus_client's default registry rather than mocking it, since these
are simple Histogram/Counter wrappers with no I/O to mock."""

from observability.dora_metrics import (
    COMPLETIONS,
    DEPLOYMENT_EVENTS,
    GATE_EVALUATIONS,
    INCIDENT_RECOVERY,
    LEAD_TIME,
    LLM_SPEND_USD,
    STEP_DURATION,
    record_workflow_completion,
)


def _counter_value(golden_path: str, status: str) -> float:
    return COMPLETIONS.labels(golden_path=golden_path, status=status)._value.get()


def _gate_counter_value(track: str, subject_type: str, subject_id: str, passed: str) -> float:
    return GATE_EVALUATIONS.labels(
        track=track, subject_type=subject_type, subject_id=subject_id, passed=passed
    )._value.get()


def _deployment_counter_value(
    track: str, subject_type: str, subject_id: str, event_type: str
) -> float:
    return DEPLOYMENT_EVENTS.labels(
        track=track, subject_type=subject_type, subject_id=subject_id, event_type=event_type
    )._value.get()


def _incident_recovery_sum(track: str, subject_type: str, subject_id: str) -> float:
    return INCIDENT_RECOVERY.labels(
        track=track, subject_type=subject_type, subject_id=subject_id
    )._sum.get()


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


def test_gate_evaluations_counter_increments() -> None:
    before = _gate_counter_value("mlops", "model", "fraud-detection:1", "true")

    GATE_EVALUATIONS.labels(
        track="mlops", subject_type="model", subject_id="fraud-detection:1", passed="true"
    ).inc()

    assert _gate_counter_value("mlops", "model", "fraud-detection:1", "true") == before + 1


def test_gate_evaluations_counter_separate_tracks() -> None:
    before_mlops = _gate_counter_value("mlops", "model", "model:1", "true")
    before_llmops = _gate_counter_value("llmops", "prompt", "prompt:1", "true")

    GATE_EVALUATIONS.labels(
        track="mlops", subject_type="model", subject_id="model:1", passed="true"
    ).inc()
    GATE_EVALUATIONS.labels(
        track="llmops", subject_type="prompt", subject_id="prompt:1", passed="true"
    ).inc()

    assert _gate_counter_value("mlops", "model", "model:1", "true") == before_mlops + 1
    assert _gate_counter_value("llmops", "prompt", "prompt:1", "true") == before_llmops + 1


def test_deployment_events_counter_increments() -> None:
    before = _deployment_counter_value("mlops", "model", "fraud-detection", "deploy")

    DEPLOYMENT_EVENTS.labels(
        track="mlops", subject_type="model", subject_id="fraud-detection", event_type="deploy"
    ).inc()

    assert _deployment_counter_value("mlops", "model", "fraud-detection", "deploy") == before + 1


def test_deployment_events_counter_separate_event_types() -> None:
    before_deploy = _deployment_counter_value("mlops", "model", "model:1", "deploy")
    before_rollback = _deployment_counter_value("mlops", "model", "model:1", "rollback")

    DEPLOYMENT_EVENTS.labels(
        track="mlops", subject_type="model", subject_id="model:1", event_type="deploy"
    ).inc()
    DEPLOYMENT_EVENTS.labels(
        track="mlops", subject_type="model", subject_id="model:1", event_type="rollback"
    ).inc()

    assert _deployment_counter_value("mlops", "model", "model:1", "deploy") == before_deploy + 1
    assert _deployment_counter_value("mlops", "model", "model:1", "rollback") == before_rollback + 1


def test_incident_recovery_histogram_observes() -> None:
    sum_before = _incident_recovery_sum("llmops", "prompt", "my-prompt")

    INCIDENT_RECOVERY.labels(track="llmops", subject_type="prompt", subject_id="my-prompt").observe(
        120.5
    )

    assert _incident_recovery_sum("llmops", "prompt", "my-prompt") == sum_before + 120.5


def test_llm_spend_gauge_sets_value() -> None:
    LLM_SPEND_USD.labels(model="claude-sonnet-5").set(12.50)
    assert LLM_SPEND_USD.labels(model="claude-sonnet-5")._value.get() == 12.50

    # Setting again should overwrite
    LLM_SPEND_USD.labels(model="claude-sonnet-5").set(15.75)
    assert LLM_SPEND_USD.labels(model="claude-sonnet-5")._value.get() == 15.75
