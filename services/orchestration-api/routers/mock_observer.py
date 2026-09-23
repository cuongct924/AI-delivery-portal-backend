"""Mock OpenChoreo observer + RCA agent — the local stand-in for the
observability plane's observer and AI RCA agent, neither of which is deployed
in the local k3d setup.

The Portal's project-page tabs (Traces, Logs, Incidents, RCA Reports) call the
observer/RCA agent directly. This router reproduces those wire contracts so the
tabs render instead of 404ing.

Data source: the same captured Golden Path runs Delivery Insights uses
(`adapters/delivery/_captured_runs.py` → mock_data/delivery_insights_runs.json,
produced by scripts/capture-delivery-insights-runs.sh). Each run becomes a
trace, its steps become log lines, and its failures become incidents + RCA
reports — so the tabs reflect real platform activity, not canned rows. When the
requested window holds no run (the tabs default to 10m, the capture is
day-grained), the full captured set is used so the tabs are never empty; with
no capture file at all, a small synthetic set is returned.

Unauthenticated on purpose — the browser calls it directly, same as
routers/delivery_insights.py.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query
from pydantic import BaseModel

from adapters.delivery._captured_runs import CapturedRun, load_captured_runs, parse_iso

router = APIRouter(tags=["mock-observer"])

_COMPONENT = "train-track-register"
_ENVIRONMENT = "development"


def _iso(minutes_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


class SearchScope(BaseModel):
    namespace: str
    project: str | None = None
    component: str | None = None
    environment: str | None = None


class QueryRequest(BaseModel):
    startTime: str | None = None
    endTime: str | None = None
    limit: int = 100
    sortOrder: str = "desc"
    searchScope: SearchScope
    logLevels: list[str] | None = None
    searchPhrase: str | None = None


def _selected_runs(request: QueryRequest) -> list[CapturedRun]:
    """Captured runs in the requested window, newest first. Falls back to the
    full captured set when the window is empty (see module docstring)."""
    runs = load_captured_runs()
    if not runs:
        return []
    if request.startTime and request.endTime:
        try:
            start = parse_iso(request.startTime)
            end = parse_iso(request.endTime)
        except ValueError:
            start = end = None  # type: ignore[assignment]
        if start is not None and end is not None:
            in_window = [r for r in runs if start <= parse_iso(r["started_at"]) < end]
            if in_window:
                runs = in_window
    runs = sorted(runs, key=lambda r: r["started_at"], reverse=request.sortOrder == "desc")
    return runs[: request.limit]


def _component(request: QueryRequest) -> str:
    return request.searchScope.component or _COMPONENT


# --- Traces ---------------------------------------------------------------


@router.post("/api/v1alpha1/traces/query")
def query_traces(request: QueryRequest) -> dict:
    runs = _selected_runs(request)
    if not runs:
        return _synthetic_traces(request)
    component = _component(request)
    traces = []
    for run in runs:
        started = parse_iso(run["started_at"])
        finished = parse_iso(run["finished_at"])
        traces.append(
            {
                "traceId": run["name"],
                "traceName": f"{run['name']} ({component})",
                "spanCount": len(run["steps"]) + 2,
                "rootSpanId": f"{run['name']}-root",
                "rootSpanName": "train-register-golden-path",
                "rootSpanKind": "SERVER",
                "startTime": started.isoformat(),
                "endTime": finished.isoformat(),
                "durationNs": int((finished - started).total_seconds() * 1_000_000_000),
                "hasErrors": run["outcome"] != "success",
            }
        )
    return {"traces": traces, "total": len(traces), "tookMs": 12}


# --- Incidents ------------------------------------------------------------


@router.post("/api/v1alpha1/incidents/query")
def query_incidents(request: QueryRequest) -> dict:
    runs = _selected_runs(request)
    if not runs:
        return _synthetic_incidents(request)
    scope = request.searchScope
    component = _component(request)
    failures = [r for r in runs if r["outcome"] != "success"]
    incidents = []
    for index, run in enumerate(failures):
        finished = parse_iso(run["finished_at"])
        # Newest failure is still active; older ones were recovered by the
        # next successful run.
        active = index == 0
        incidents.append(
            {
                "incidentId": f"INC-{run['name'][-6:]}",
                "alertId": f"alert-{run['name'][-6:]}",
                "status": "active" if active else "resolved",
                "description": f"Golden Path run {run['name']} failed at the train step",
                "notes": "Auto-opened from the run's failure event.",
                "timestamp": finished.isoformat(),
                "triggeredAt": finished.isoformat(),
                **({} if active else {"resolvedAt": finished.isoformat()}),
                "incidentTriggerAiRca": True,
                "incidentTriggerAiCostAnalysis": False,
                "projectName": scope.project,
                "componentName": component,
                "environmentName": scope.environment,
                "namespaceName": scope.namespace,
            }
        )
    return {"incidents": incidents, "total": len(incidents)}


# --- Runtime logs ---------------------------------------------------------


@router.post("/api/v1/logs/query")
def query_logs(request: QueryRequest) -> dict:
    runs = _selected_runs(request)
    if not runs:
        return _synthetic_logs(request)
    scope = request.searchScope
    component = _component(request)
    levels = request.logLevels or ["INFO", "WARN", "ERROR"]
    logs = []
    for run in runs:
        failed = run["outcome"] != "success"
        logs.append(
            {
                "timestamp": run["started_at"],
                "log": f"workflow {run['name']} started",
                "level": "INFO",
                "metadata": _log_metadata(scope, component),
            }
        )
        for step in run["steps"]:
            logs.append(
                {
                    "timestamp": step["started_at"],
                    "log": f"step {step['name']} started",
                    "level": "INFO",
                    "metadata": _log_metadata(scope, component),
                }
            )
            logs.append(
                {
                    "timestamp": step["finished_at"],
                    "log": (
                        f"step {step['name']} failed"
                        if failed
                        else f"step {step['name']} completed"
                    ),
                    "level": "ERROR" if failed else "INFO",
                    "metadata": _log_metadata(scope, component),
                }
            )
    logs = [entry for entry in logs if entry["level"] in levels]
    if request.searchPhrase:
        phrase = request.searchPhrase.lower()
        logs = [entry for entry in logs if phrase in entry["log"].lower()]
    return {"logs": logs[: request.limit], "total": len(logs), "tookMs": 8}


def _log_metadata(scope: SearchScope, component: str) -> dict:
    return {
        "componentName": component,
        "projectName": scope.project,
        "environmentName": scope.environment,
        "namespaceName": scope.namespace,
    }


# --- RCA reports ----------------------------------------------------------


@router.get("/api/v1/rca-agent/reports")
def list_rca_reports(
    namespace: str = Query(...),
    project: str = Query(...),
    environment: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100),
) -> dict:
    runs = load_captured_runs()
    failures = [r for r in runs if r["outcome"] != "success"]
    failures.sort(key=lambda r: r["started_at"], reverse=True)
    reports = [_rca_summary(run, project) for run in failures[:limit]]
    if status:
        reports = [r for r in reports if r["status"] == status]
    return {"reports": reports, "totalCount": len(reports)}


def _rca_summary(run: CapturedRun, project: str) -> dict:
    return {
        "alertId": f"alert-{run['name'][-6:]}",
        "reportId": f"rca-{run['name']}",
        "timestamp": run["finished_at"],
        "summary": (
            f"Golden Path run **{run['name']}** failed at the "
            f"`{run['steps'][0]['name'] if run['steps'] else 'train'}` step. "
            f"The run never reached the register step, so no model version was "
            f"produced. Re-running after the fix is the recovery path."
        ),
        "status": "completed",
    }


@router.get("/api/v1/rca-agent/reports/{report_id}")
def get_rca_report(report_id: str) -> dict:
    run_name = report_id.removeprefix("rca-")
    run = next((r for r in load_captured_runs() if r["name"] == run_name), None)
    if run is None:
        run = {
            "name": run_name,
            "started_at": _iso(60),
            "finished_at": _iso(55),
            "outcome": "failure",
            "steps": [{"name": "train", "started_at": _iso(60), "finished_at": _iso(55)}],
        }
    step = run["steps"][0]["name"] if run["steps"] else "train"
    return {
        "alertId": f"alert-{run['name'][-6:]}",
        "reportId": report_id,
        "timestamp": run["finished_at"],
        "status": "completed",
        "report": {
            "alert_context": {
                "alert_id": f"alert-{run['name'][-6:]}",
                "alert_name": "Golden Path run failed",
                "alert_description": f"Run {run['name']} finished with outcome {run['outcome']}",
                "severity": "warning",
                "triggered_at": run["finished_at"],
                "trigger_value": 1.0,
                "source_type": "log",
                "condition": {
                    "window": "1m",
                    "interval": "1m",
                    "operator": "==",
                    "threshold": 1.0,
                },
                "component": _COMPONENT,
                "project": "default",
                "environment": _ENVIRONMENT,
            },
            "summary": (
                f"The `{step}` step of **{run['name']}** failed, so the run "
                f"stopped before registering a model."
            ),
            "result": {
                "type": "root_cause_identified",
                "root_causes": [
                    {
                        "summary": f"The {step} step raised an error",
                        "confidence": "high",
                        "analysis": (
                            "The step's container exited non-zero; the workflow "
                            "marked the run failed and skipped the register step."
                        ),
                        "supporting_findings": [
                            {
                                "observation": f"step {step} failed",
                                "component": _COMPONENT,
                                "time_range": {
                                    "start": run["started_at"],
                                    "end": run["finished_at"],
                                },
                                "evidence": {
                                    "type": "log",
                                    "log_lines": [
                                        {
                                            "timestamp": run["finished_at"],
                                            "level": "ERROR",
                                            "log": f"step {step} failed",
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
                "timeline": [
                    {
                        "timestamp": run["started_at"],
                        "event": "Run started",
                        "description": run["name"],
                    },
                    {
                        "timestamp": run["finished_at"],
                        "event": "Run failed",
                        "description": f"step {step}",
                    },
                ],
                "recommendations": {
                    "immediate": ["Inspect the failed step's logs and re-run"],
                    "long_term": ["Add a pre-flight check for the failing input"],
                },
            },
            "investigation_path": [
                {
                    "action": "Read the run's step outcomes",
                    "outcome": f"step {step} failed",
                    "rationale": "Locates the failing stage",
                }
            ],
        },
    }


# --- Synthetic fallback (no capture file) ---------------------------------


def _synthetic_traces(request: QueryRequest) -> dict:
    component = _component(request)
    traces = [
        {
            "traceId": f"trace-{component}-{i}",
            "traceName": f"train-register-golden-path ({component})",
            "spanCount": 3,
            "rootSpanId": f"span-{i}",
            "rootSpanName": "train-register-golden-path",
            "rootSpanKind": "SERVER",
            "startTime": _iso(30 - i * 5),
            "endTime": _iso(29 - i * 5),
            "durationNs": (120 + i * 40) * 1_000_000,
            "hasErrors": i == 1,
        }
        for i in range(3)
    ]
    return {"traces": traces, "total": len(traces), "tookMs": 12}


def _synthetic_incidents(request: QueryRequest) -> dict:
    scope = request.searchScope
    component = _component(request)
    incidents = [
        {
            "incidentId": f"INC-{scope.project}-1",
            "alertId": f"alert-{scope.project}-1",
            "status": "active",
            "description": f"Golden Path run failed on {component}",
            "timestamp": _iso(45),
            "triggeredAt": _iso(45),
            "incidentTriggerAiRca": True,
            "incidentTriggerAiCostAnalysis": False,
            "projectName": scope.project,
            "componentName": component,
            "environmentName": scope.environment,
            "namespaceName": scope.namespace,
        }
    ]
    return {"incidents": incidents, "total": len(incidents)}


def _synthetic_logs(request: QueryRequest) -> dict:
    scope = request.searchScope
    component = _component(request)
    logs = [
        {
            "timestamp": _iso(5 + i * 3),
            "log": message,
            "level": level,
            "metadata": _log_metadata(scope, component),
        }
        for i, (level, message) in enumerate(
            [
                ("INFO", "workflow started"),
                ("ERROR", "step train failed"),
            ]
        )
    ]
    return {"logs": logs, "total": len(logs), "tookMs": 8}
