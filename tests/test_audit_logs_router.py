"""services/orchestration-api/routers/audit_logs.py — the mock observer audit
contract the Audit Logs page calls. Pure synthetic data, so the functions are
called directly."""

from typing import Any, cast

from audit.events import record_audit_event
from routers.audit_logs import (
    AuditLogFilterValuesRequest,
    AuditLogsQueryRequest,
    query_audit_log_filter_values,
    query_audit_logs,
)

_WIDE: dict[str, Any] = {
    "startTime": "2000-01-01T00:00:00+00:00",
    "endTime": "2100-01-01T00:00:00+00:00",
}

_WINDOW: dict[str, Any] = {
    "startTime": "2026-07-01T00:00:00+00:00",
    "endTime": "2026-07-02T00:00:00+00:00",
}


def _query(**over: Any) -> dict[str, Any]:
    return cast(dict[str, Any], query_audit_logs(AuditLogsQueryRequest(**_WINDOW, **over)))


def _filter_values(**over: Any) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        query_audit_log_filter_values(
            AuditLogFilterValuesRequest(query=AuditLogsQueryRequest(**_WINDOW), **over)
        ),
    )


def test_query_returns_records_with_the_expected_shape() -> None:
    response = _query()
    assert response["total"] > 0
    record = response["records"][0]
    assert record["schema_version"] == "1.0"
    assert "actor" in record and "action" in record and "result" in record


def test_query_is_deterministic() -> None:
    first = _query()
    second = _query()
    assert [r["event_id"] for r in first["records"]] == [r["event_id"] for r in second["records"]]


def test_query_filters_by_result() -> None:
    response = _query(result=["denied"])
    assert response["total"] > 0
    assert all(r["result"] == "denied" for r in response["records"])


def test_query_respects_limit_and_sort_order() -> None:
    response = _query(limit=5, sortOrder="asc")
    assert len(response["records"]) == 5
    times = [r["event_time"] for r in response["records"]]
    assert times == sorted(times)


def test_query_includes_timeline_when_requested() -> None:
    response = _query(includeTimeline=True, timelineInterval="1h")
    assert "timeline" in response
    assert response["timeline"]["interval"] == "1h"


def test_filter_values_lists_values_with_counts() -> None:
    response = _filter_values(filter="result")
    values = {v["value"] for v in response["values"]}
    assert "success" in values
    assert response["totalValues"] >= len(values)


def test_filter_values_narrows_by_search() -> None:
    response = _filter_values(filter="action", valueSearch="deploy")
    assert all("deploy" in v["value"] for v in response["values"])


def test_recorded_events_appear_in_the_query() -> None:
    record_audit_event(
        action="training.trigger",
        resource={"name": "fraud-detection", "type": "Model"},
    )
    response = cast(dict[str, Any], query_audit_logs(AuditLogsQueryRequest(**_WIDE)))
    # The synthetic baseline also has a training.trigger action, so match on the
    # resource name only the recorded event carries.
    match = [
        r for r in response["records"] if r.get("resource", {}).get("name") == "fraud-detection"
    ]
    assert match, "expected the recorded event to appear in the trail"
    assert match[0]["producer"] == "orchestration-api"
    assert match[0]["action"] == "training.trigger"
