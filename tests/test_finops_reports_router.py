"""services/orchestration-api/routers/finops_reports.py — calls the route
functions directly, same pattern as the other router tests."""

import pytest
from fastapi import HTTPException
from routers.finops_reports import get_report, list_reports


def test_list_reports_returns_one_per_component() -> None:
    response = list_reports(
        namespace="default", project="default", environment=None, status=None, limit=100
    )

    assert response["totalCount"] == 3
    assert {r["component"] for r in response["reports"]} == {
        "orchestration-api",
        "mlops-golden-paths-server",
        "portal-assistant",
    }
    # The list omits the heavy nested report body.
    assert all("report" not in r for r in response["reports"])


def test_list_reports_filters_by_status() -> None:
    response = list_reports(
        namespace="default", project="default", environment=None, status="failed", limit=100
    )

    assert response["reports"] == []


def test_list_reports_respects_limit() -> None:
    response = list_reports(
        namespace="default", project="default", environment=None, status=None, limit=1
    )

    assert len(response["reports"]) == 1


def test_get_report_returns_the_nested_report_body() -> None:
    report = get_report("finops-default-orchestration-api")

    assert report["component"] == "orchestration-api"
    assert report["report"]["overprovisioning"]["is_overprovisioned"] is True
    assert report["report"]["recommended_actions"]


def test_get_report_raises_404_for_unknown_id() -> None:
    with pytest.raises(HTTPException) as exc:
        get_report("finops-nope")

    assert exc.value.status_code == 404
