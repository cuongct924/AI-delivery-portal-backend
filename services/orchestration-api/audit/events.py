"""The one helper every golden-path request calls to record an audit event.

Kept deliberately small and side-effect-only: a request that mutates state
calls `record_audit_event(...)` and moves on. Recording never raises — a trail
write failure must not fail the action that produced it.

File-backed for the same reason the cost ledger is: a demo needs the trail to
survive a restart without a database, and the write volume is tiny (one entry
per golden-path run, not per request).
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PATH = Path(os.getenv("AUDIT_LOG_PATH", ".state/audit-log.json"))

# The issuer every local Thunder token carries; kept on the actor so the page's
# actor.issuer filter has a value to match.
_ISSUER = "https://thunder.openchoreo.localhost:8080"


class AuditEvent(TypedDict):
    schema_version: str
    event_id: str
    event_time: str
    actor: dict[str, object]
    action: str
    category: str
    result: str
    producer: str
    surface: str
    operation_id: str
    resource: dict[str, object]
    metadata: dict[str, object]


def _read() -> list[dict[str, object]]:
    if not _PATH.exists():
        return []
    try:
        return json.loads(_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def record_audit_event(
    *,
    action: str,
    resource: dict[str, object] | None = None,
    result: str = "success",
    category: str = "management",
    surface: str = "rest",
    actor_id: str = "orchestration-api",
    actor_type: str = "service-account",
    metadata: dict[str, object] | None = None,
) -> None:
    """Append one audit event. `action` is the semantic name policy is written
    against (e.g. "training.trigger"); `result` is success | failure | denied."""
    event: AuditEvent = {
        "schema_version": "1.0",
        "event_id": f"evt-{uuid4().hex[:12]}",
        "event_time": datetime.now(UTC).isoformat(),
        "actor": {"type": actor_type, "id": actor_id, "issuer": _ISSUER},
        "action": action,
        "category": category,
        "result": result,
        "producer": "orchestration-api",
        "surface": surface,
        "operation_id": action,
        "resource": resource or {},
        "metadata": metadata or {},
    }
    try:
        with _LOCK:
            events = _read()
            events.append(dict(event))
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            _PATH.write_text(json.dumps(events, indent=2))
    except Exception as exc:  # noqa: BLE001 — never fail the caller
        logger.warning("Failed to record audit event %s: %s", action, exc)


def read_audit_events() -> list[dict[str, object]]:
    """Every recorded event, oldest first."""
    with _LOCK:
        return _read()
