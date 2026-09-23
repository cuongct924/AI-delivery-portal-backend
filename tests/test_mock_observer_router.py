"""services/orchestration-api/routers/mock_observer.py — calls the route
functions directly, same pattern as the other router tests. The router derives
its rows from the captured Golden Path runs, so assertions are relative to that
file rather than hard-coded counts."""

from routers.mock_observer import (
    QueryRequest,
    SearchScope,
    get_rca_report,
    list_rca_reports,
    query_incidents,
    query_logs,
    query_traces,
)

from adapters.delivery._captured_runs import load_captured_runs

_RUNS = load_captured_runs()
_FAILURES = [r for r in _RUNS if r["outcome"] != "success"]


def _request(**overrides) -> QueryRequest:
    scope = SearchScope(
        namespace="default", project="telco-fraud-detection", environment="development"
    )
    return QueryRequest(searchScope=scope, **overrides)


def test_query_traces_returns_one_trace_per_captured_run() -> None:
    response = query_traces(_request())

    assert response["total"] == len(_RUNS)
    assert {t["traceId"] for t in response["traces"]} == {r["name"] for r in _RUNS}


def test_query_traces_marks_failed_runs_as_errors() -> None:
    response = query_traces(_request())

    errored = {t["traceId"] for t in response["traces"] if t["hasErrors"]}
    assert errored == {r["name"] for r in _FAILURES}


def test_query_incidents_returns_one_per_failed_run() -> None:
    response = query_incidents(_request())

    assert response["total"] == len(_FAILURES)
    # Newest failure is active, the rest resolved.
    assert response["incidents"][0]["status"] == "active"
    assert all(i["status"] == "resolved" for i in response["incidents"][1:])


def test_query_logs_filters_by_level() -> None:
    response = query_logs(_request(logLevels=["ERROR"]))

    assert response["total"] == len(_FAILURES)
    assert all(entry["level"] == "ERROR" for entry in response["logs"])


def test_list_rca_reports_returns_one_per_failed_run() -> None:
    response = list_rca_reports(
        namespace="default",
        project="telco-fraud-detection",
        environment="development",
        status=None,
        limit=100,
    )

    assert response["totalCount"] == len(_FAILURES)


def test_get_rca_report_returns_the_nested_report() -> None:
    report = get_rca_report(f"rca-{_FAILURES[0]['name']}")

    assert report["status"] == "completed"
    assert report["report"]["result"]["type"] == "root_cause_identified"
