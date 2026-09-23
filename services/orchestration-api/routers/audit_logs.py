"""Mock OpenChoreo observer audit-trail API for the Audit Logs page.

The Portal's Audit Logs page calls the OpenChoreo observer directly
(`POST /api/v1alpha1/audit-logs/query` and `/filter-values`). Without a real
observability plane those calls 404, so this router reproduces the contract
with deterministic synthetic records — the same role
routers/observer_costs.py plays for the cost API.

Records keep the observer's snake_case field names and nesting, so the page's
column/filter paths resolve unchanged.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Final

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1alpha1/audit-logs", tags=["audit-logs"])

_ACTORS: Final[list[dict[str, object]]] = [
    {
        "type": "user",
        "id": "cuongct",
        "issuer": "https://thunder.openchoreo.localhost:8080",
        "session_id": "sess-4f2a",
        "entitlements": {"groups": ["platform-engineers"]},
    },
    {
        "type": "service-account",
        "id": "orchestration-api",
        "issuer": "https://thunder.openchoreo.localhost:8080",
    },
    {
        "type": "user",
        "id": "mlops-bot",
        "issuer": "https://thunder.openchoreo.localhost:8080",
        "session_id": "sess-9c11",
        "entitlements": {"groups": ["mlops-team"]},
    },
]

_ACTIONS: Final[list[str]] = [
    "component.create",
    "component.update",
    "release.create",
    "releasebinding.update",
    "model.deploy",
    "model.rollback",
    "prompt.activate",
    "rag.ingest",
    "training.trigger",
    "monitoring.setup",
]

_RESOURCES: Final[list[dict[str, object]]] = [
    {
        "type": "Component",
        "namespace": "default",
        "project": "ai-delivery-portal",
        "component": "orchestration-api",
        "name": "orchestration-api",
    },
    {
        "type": "Component",
        "namespace": "default",
        "project": "telco-fraud-detection",
        "component": "serving",
        "name": "serving",
    },
    {
        "type": "ReleaseBinding",
        "namespace": "default",
        "environment": "default/production",
        "project": "telco-fraud-detection",
        "component": "serving",
        "name": "serving-production",
    },
    {
        "type": "Component",
        "namespace": "default",
        "project": "customer-segmentation",
        "component": "training",
        "name": "training",
    },
]

_RESULTS: Final[list[str]] = ["success", "success", "success", "failure", "denied"]
_CATEGORIES: Final[list[str]] = ["management", "authorization", "access"]
_SURFACES: Final[list[str]] = ["rest", "rest", "mcp"]
_PRODUCERS: Final[list[str]] = ["openchoreo-api", "orchestration-api"]

_RECORD_COUNT: Final[int] = 60


class AuditLogsQueryRequest(BaseModel):
    startTime: str
    endTime: str
    limit: int = 100
    sortOrder: str = "desc"
    includeTimeline: bool = False
    timelineInterval: str | None = None
    actor: dict[str, list[str]] | None = None
    resource: dict[str, list[str]] | None = None
    action: list[str] | None = None
    category: list[str] | None = None
    result: list[str] | None = None
    producer: list[str] | None = None
    surface: list[str] | None = None
    operation_id: list[str] | None = None
    request_id: list[str] | None = None
    event_id: list[str] | None = None
    source_ip: list[str] | None = None
    user_agent: list[str] | None = None
    searchPhrase: str | None = None


class AuditLogFilterValuesRequest(BaseModel):
    query: AuditLogsQueryRequest
    filter: str
    valueSearch: str | None = None
    maxValues: int = 50


def _seed(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16)


def _parse(value: str, fallback: datetime) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return fallback


def _generate(start: datetime, end: datetime) -> list[dict[str, object]]:
    """Deterministic records spread across the window, so repeated queries and
    the filter-values call all see the same trail."""
    span = (end - start).total_seconds()
    if span <= 0:
        return []
    records: list[dict[str, object]] = []
    for i in range(_RECORD_COUNT):
        seed = _seed(start.isoformat(), str(i))
        ts = start + timedelta(seconds=span * i / _RECORD_COUNT)
        actor = _ACTORS[seed % len(_ACTORS)]
        action = _ACTIONS[seed % len(_ACTIONS)]
        resource = _RESOURCES[seed % len(_RESOURCES)]
        result = _RESULTS[seed % len(_RESULTS)]
        category = _CATEGORIES[seed % len(_CATEGORIES)]
        surface = _SURFACES[seed % len(_SURFACES)]
        producer = _PRODUCERS[seed % len(_PRODUCERS)]
        event_id = f"evt-{seed:08x}"
        records.append(
            {
                "schema_version": "1.0",
                "event_id": event_id,
                "event_time": ts.isoformat(),
                "actor": actor,
                "action": action,
                "category": category,
                "result": result,
                "request_id": f"req-{seed:08x}",
                "source_ip": f"10.0.{seed % 255}.{seed % 200}",
                "user_agent": "backstage-portal/1.0",
                "producer": producer,
                "surface": surface,
                "operation_id": action,
                "http": {"method": "POST", "path": f"/api/v1/{action.replace('.', '/')}"},
                "resource": resource,
                "metadata": {"seed": seed},
                "collector": {
                    "namespaceName": "openchoreo-observability",
                    "podName": f"collector-{seed % 3}",
                    "containerName": "collector",
                },
            }
        )
    return records


def _matches(record: dict[str, object], request: AuditLogsQueryRequest) -> bool:
    def any_of(values: list[str] | None, actual: object) -> bool:
        return not values or str(actual) in values

    actor = record["actor"]
    resource = record["resource"] or {}
    assert isinstance(actor, dict) and isinstance(resource, dict)

    if request.actor:
        for key, values in request.actor.items():
            if not any_of(values, actor.get(key)):
                return False
    if request.resource:
        for key, values in request.resource.items():
            if not any_of(values, resource.get(key)):
                return False
    if not any_of(request.action, record["action"]):
        return False
    if not any_of(request.category, record["category"]):
        return False
    if not any_of(request.result, record["result"]):
        return False
    if not any_of(request.producer, record["producer"]):
        return False
    if not any_of(request.surface, record["surface"]):
        return False
    if not any_of(request.operation_id, record["operation_id"]):
        return False
    if not any_of(request.request_id, record["request_id"]):
        return False
    if not any_of(request.event_id, record["event_id"]):
        return False
    if not any_of(request.source_ip, record["source_ip"]):
        return False
    if not any_of(request.user_agent, record["user_agent"]):
        return False
    if request.searchPhrase:
        phrase = request.searchPhrase.lower()
        haystack = " ".join(
            [str(record["action"]), str(actor.get("id")), str(resource.get("name"))]
        ).lower()
        if phrase not in haystack:
            return False
    return True


def _filtered(request: AuditLogsQueryRequest) -> list[dict[str, object]]:
    now = datetime.now(UTC)
    start = _parse(request.startTime, now - timedelta(days=1))
    end = _parse(request.endTime, now)
    records = [r for r in _generate(start, end) if _matches(r, request)]
    records.sort(key=lambda r: str(r["event_time"]), reverse=request.sortOrder != "asc")
    return records


def _timeline(records: list[dict[str, object]], interval: str) -> dict[str, object]:
    unit = interval[-1:] if interval else "h"
    amount = int(interval[:-1]) if interval[:-1].isdigit() else 1
    step = {
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
        "d": timedelta(days=amount),
        "w": timedelta(weeks=amount),
    }.get(unit, timedelta(hours=1))

    buckets: dict[str, int] = {}
    for record in records:
        ts = _parse(str(record["event_time"]), datetime.now(UTC))
        bucket = ts.replace(
            minute=0 if step >= timedelta(hours=1) else ts.minute,
            second=0,
            microsecond=0,
        )
        key = bucket.isoformat()
        buckets[key] = buckets.get(key, 0) + 1
    return {
        "interval": interval or "1h",
        "buckets": [{"startTime": k, "total": v} for k, v in sorted(buckets.items())],
    }


@router.post("/query")
def query_audit_logs(request: AuditLogsQueryRequest) -> dict[str, object]:
    records = _filtered(request)
    total = len(records)
    limited = records[: max(0, request.limit)]
    response: dict[str, object] = {
        "records": limited,
        "total": total,
        "tookMs": 3,
    }
    if request.includeTimeline:
        response["timeline"] = _timeline(records, request.timelineInterval or "1h")
    return response


def _value_at(record: dict[str, object], path: str) -> list[str]:
    """Resolve a filter path (e.g. "actor.id") to the record's value(s)."""
    node: object = record
    for part in path.split("."):
        if not isinstance(node, dict):
            return []
        node = node.get(part)
    if node is None:
        return []
    if isinstance(node, list):
        return [str(v) for v in node]
    if isinstance(node, dict):
        # e.g. actor.entitlements -> flatten its values
        return [
            str(v)
            for values in node.values()
            for v in (values if isinstance(values, list) else [values])
        ]
    return [str(node)]


@router.post("/filter-values")
def query_audit_log_filter_values(
    request: AuditLogFilterValuesRequest,
) -> dict[str, object]:
    records = _filtered(request.query)
    counts: dict[str, int] = {}
    for record in records:
        for value in _value_at(record, request.filter):
            counts[value] = counts.get(value, 0) + 1

    values = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if request.valueSearch:
        needle = request.valueSearch.lower()
        values = [(v, c) for v, c in values if needle in v.lower()]
    total_values = len(values)
    limited = values[: max(0, request.maxValues)]
    return {
        "filter": request.filter,
        "values": [{"value": v, "count": c} for v, c in limited],
        "totalValues": total_values,
        "tookMs": 2,
    }
