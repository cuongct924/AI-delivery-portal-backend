"""adapters/delivery/deployment_event_store.py — exercises
SqliteDeploymentEventStore against a real sqlite file on tmp_path, same
convention as tests/test_prediction_log_adapter.py."""

from datetime import UTC, datetime

from adapters.delivery.deployment_event_store import SqliteDeploymentEventStore

_START = datetime(2026, 8, 1, tzinfo=UTC)
_END = datetime(2026, 8, 3, tzinfo=UTC)


def _store(tmp_path) -> SqliteDeploymentEventStore:
    return SqliteDeploymentEventStore(path=str(tmp_path / "deployment_events.db"))


def test_list_events_returns_empty_list_when_nothing_recorded(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.list_events(_START, _END, 100, "desc") == []


def test_record_event_then_list_events_round_trips(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_event(
        name="train-track-register-abc12",
        change_type="model",
        project_name="fraud-detection",
        component_name="train-track-register",
        environment_name="development",
        outcome="success",
        started_at="2026-08-01T10:00:00",
        finished_at="2026-08-01T11:00:00",
        steps=[{"name": "train", "phase": None, "started_at": None, "finished_at": None}],
    )

    [event] = store.list_events(_START, _END, 100, "desc")
    assert event["name"] == "train-track-register-abc12"
    assert event["changeType"] == "model"
    assert event["projectName"] == "fraud-detection"
    assert event["outcome"] == "success"
    assert event["steps"] == [
        {"name": "train", "phase": None, "started_at": None, "finished_at": None}
    ]


def test_record_event_tolerates_missing_steps(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_event(
        name="promote-fraud-detection-1",
        change_type="model",
        project_name="fraud-detection",
        component_name="serving",
        environment_name="staging",
        outcome="success",
        started_at="2026-08-01T10:00:00",
        finished_at="2026-08-01T10:00:00",
    )

    [event] = store.list_events(_START, _END, 100, "desc")
    assert event["steps"] is None


def test_list_events_excludes_events_outside_the_window(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_event(
        name="before-window",
        change_type="model",
        project_name="p",
        component_name="c",
        environment_name="development",
        outcome="success",
        started_at="2026-07-01T00:00:00",
        finished_at="2026-07-01T00:00:00",
    )
    store.record_event(
        name="in-window",
        change_type="model",
        project_name="p",
        component_name="c",
        environment_name="development",
        outcome="success",
        started_at="2026-08-02T00:00:00",
        finished_at="2026-08-02T00:00:00",
    )

    [event] = store.list_events(_START, _END, 100, "desc")
    assert event["name"] == "in-window"


def test_list_events_respects_limit_and_sort_order(tmp_path) -> None:
    store = _store(tmp_path)
    for day in (1, 2):
        store.record_event(
            name=f"event-{day}",
            change_type="model",
            project_name="p",
            component_name="c",
            environment_name="development",
            outcome="success",
            started_at=f"2026-08-0{day}T00:00:00",
            finished_at=f"2026-08-0{day}T00:00:00",
        )

    descending = store.list_events(_START, _END, 100, "desc")
    assert [e["name"] for e in descending] == ["event-2", "event-1"]

    ascending = store.list_events(_START, _END, 100, "asc")
    assert [e["name"] for e in ascending] == ["event-1", "event-2"]

    limited = store.list_events(_START, _END, 1, "desc")
    assert len(limited) == 1


def test_state_persists_across_separate_store_instances(tmp_path) -> None:
    path = str(tmp_path / "deployment_events.db")
    SqliteDeploymentEventStore(path=path).record_event(
        name="e1",
        change_type="prompt",
        project_name="p",
        component_name="prompt",
        environment_name="development",
        outcome="success",
        started_at="2026-08-01T00:00:00",
        finished_at="2026-08-01T00:00:00",
    )
    reloaded = SqliteDeploymentEventStore(path=path)
    assert len(reloaded.list_events(_START, _END, 100, "desc")) == 1
