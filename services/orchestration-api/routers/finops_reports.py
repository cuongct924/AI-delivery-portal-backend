"""FinOps cost-analysis reports API — a local stand-in for the OpenChoreo
FinOps agent (`spec.finOpsAgentURL` on the observability plane), which isn't
deployed in the local k3d setup.

The Portal's Cost Analysis tab calls the FinOps agent directly
(`/api/v1alpha1/reports` and `/api/v1alpha1/reports/{id}`). This router
reproduces that contract with deterministic mock reports, so the page renders
end to end. Point the frontend's FinOps URL at orchestration-api (frontend
config `openchoreo.observability.mockObserverUrl`, resolved as finopsAgentUrl
by the observability backend) and the tab works without the real agent.

Unauthenticated on purpose — the browser calls it directly, same as
routers/delivery_insights.py.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/v1alpha1/reports", tags=["finops-reports"])

# One report per (component, day-offset) — enough rows to exercise the table,
# the status filter, and the detail view without a real agent.
_COMPONENTS = ["orchestration-api", "mlops-golden-paths-server", "portal-assistant"]


def _iso(days_ago: int, hour: int = 9) -> str:
    day = datetime.now(UTC) - timedelta(days=days_ago)
    return day.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


def _summary(component: str, overprovisioned: bool, saving: float) -> str:
    if overprovisioned:
        return (
            f"**{component}** is over-provisioned: CPU/memory requests sit far "
            f"above observed usage. Right-sizing the release binding would save "
            f"about **${saving:.2f}/month** with no throughput impact."
        )
    return (
        f"**{component}** is right-sized — observed CPU/memory usage tracks the "
        f"configured requests closely. No action needed this period."
    )


def _report(report_id: str, namespace: str, project: str, component: str, days_ago: int) -> dict:
    overprovisioned = component != "portal-assistant"
    saving = 12.66 if component == "orchestration-api" else 11.09
    actual = 90.16 if component == "orchestration-api" else 42.5
    budget = actual * 1.4
    period_start = (datetime.now(UTC) - timedelta(days=30)).date()
    period_end = datetime.now(UTC).date()
    return {
        "reportId": report_id,
        "namespace": namespace,
        "project": project,
        "environment": "development",
        "component": component,
        "timestamp": _iso(days_ago),
        "summary": _summary(component, overprovisioned, saving),
        "status": "completed",
        "report": {
            "component": component,
            "namespace": namespace,
            "project": project,
            "analysis_period": f"{period_start} to {period_end}",
            "budgeted_cost": {
                "total_cost": round(budget, 2),
                "currency": "USD",
                "is_estimated": True,
            },
            "actual_cost": {
                "total_cost": round(actual, 2),
                "currency": "USD",
                "is_estimated": False,
            },
            "resource_metrics": {
                "cpu_request": "1000m",
                "cpu_limit": "2000m",
                "cpu_actual_avg": "180m",
                "cpu_actual_peak": "640m",
                "memory_request": "2Gi",
                "memory_limit": "4Gi",
                "memory_actual_avg": "410Mi",
                "memory_actual_peak": "1.1Gi",
                "data_available": True,
            },
            "overprovisioning": {
                "is_overprovisioned": overprovisioned,
                "cpu_utilization_pct": 18.0 if overprovisioned else 72.0,
                "memory_utilization_pct": 20.5 if overprovisioned else 68.0,
                "analysis": (
                    "Sustained CPU and memory usage are well below the configured "
                    "requests across the whole period."
                    if overprovisioned
                    else "Usage is close to the configured requests."
                ),
                "recommendation": (
                    {
                        "cpu_request": "250m",
                        "cpu_limit": "750m",
                        "memory_request": "512Mi",
                        "memory_limit": "1Gi",
                        "rationale": "Covers the p95 observed usage with headroom.",
                        "release_binding": f"{component}-development",
                    }
                    if overprovisioned
                    else None
                ),
            },
            "summary": _summary(component, overprovisioned, saving),
            "investigation_path": [
                {
                    "action": "Read CPU/memory requests and limits from the release binding",
                    "outcome": "Requests are 5x the p95 observed usage",
                    "rationale": "Baseline for the over-provisioning check",
                },
                {
                    "action": "Compare against 30 days of Prometheus usage",
                    "outcome": "p95 CPU 640m, p95 memory 1.1Gi",
                    "rationale": "Confirms the headroom is unused",
                },
            ],
            "recommended_actions": (
                [
                    {
                        "description": "Right-size the development release binding",
                        "rationale": f"Saves ~${saving:.2f}/month",
                        "status": "revised",
                        "change": {
                            "release_binding": f"{component}-development",
                            "fields": [
                                {"json_pointer": "/cpu/request", "value": "250m"},
                                {"json_pointer": "/memory/request", "value": "512Mi"},
                            ],
                        },
                    }
                ]
                if overprovisioned
                else []
            ),
        },
    }


def _all_reports(namespace: str, project: str) -> list[dict]:
    reports: list[dict] = []
    for i, component in enumerate(_COMPONENTS):
        reports.append(_report(f"finops-{project}-{component}", namespace, project, component, i))
    return reports


@router.get("")
def list_reports(
    namespace: str = Query(...),
    project: str = Query(...),
    environment: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100),
) -> dict:
    reports = _all_reports(namespace, project)
    if environment:
        reports = [r for r in reports if r["environment"] == environment]
    if status:
        reports = [r for r in reports if r["status"] == status]
    reports = reports[:limit]
    return {
        "reports": [{k: v for k, v in r.items() if k != "report"} for r in reports],
        "totalCount": len(reports),
    }


@router.get("/{report_id}")
def get_report(report_id: str) -> dict:
    for project in ("default", "platform", "telco-fraud-detection", "ai-delivery-portal"):
        for report in _all_reports("default", project):
            if report["reportId"] == report_id:
                return report
    raise HTTPException(404, f"FinOps report {report_id!r} not found")
