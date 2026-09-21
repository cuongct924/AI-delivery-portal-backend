"""SQLite-backed store for real Delivery Insights deployment events — same
"local file, no server to run" precedent as
adapters/ai_platform/prediction_log_adapter.py's SqlitePredictionLogAdapter.

Written by routers/models.py|rag.py|prompts.py at each real training
completion / model promote-rollback / prompt-rag-index activate event; read
by PrometheusDeliveryObserverAdapter.query_deployments(). This is what lets
the Delivery Insights deployment table need no manual
scripts/capture-delivery-insights-runs.sh step for new data going forward.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from adapters.delivery._captured_runs import parse_iso
from adapters.delivery.interfaces import ChangeType, WorkflowStepTiming

_SCHEMA = """
CREATE TABLE IF NOT EXISTS deployment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    change_type TEXT NOT NULL,
    project_name TEXT NOT NULL,
    component_name TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    outcome TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    steps_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_deployment_events_started_at ON deployment_events (started_at);
"""


class DeploymentEventRecord(TypedDict):
    name: str
    changeType: ChangeType
    projectName: str
    componentName: str
    environmentName: str
    outcome: Literal["success", "failed", "in_progress"]
    startedAt: str
    finishedAt: str
    # WorkflowStepTiming reused as-is (not a narrower local type) so
    # WorkflowStatus["steps"] can be passed straight through from
    # routers/models.py without repackaging.
    steps: list[WorkflowStepTiming] | None


class SqliteDeploymentEventStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(
            path or os.getenv("DEPLOYMENT_EVENT_LOG_PATH", ".state/deployment_events.db")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record_event(
        self,
        name: str,
        change_type: ChangeType,
        project_name: str,
        component_name: str,
        environment_name: str,
        outcome: Literal["success", "failed", "in_progress"],
        started_at: str,
        finished_at: str,
        steps: list[WorkflowStepTiming] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO deployment_events "
                "(name, change_type, project_name, component_name, environment_name, "
                "outcome, started_at, finished_at, steps_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    change_type,
                    project_name,
                    component_name,
                    environment_name,
                    outcome,
                    started_at,
                    finished_at,
                    json.dumps(steps) if steps else None,
                ),
            )

    def list_events(
        self, start: datetime, end: datetime, limit: int, sort_order: Literal["asc", "desc"]
    ) -> list[DeploymentEventRecord]:
        """Filters/sorts in Python rather than in SQL — stored `started_at`
        values are naive (routers write `datetime.now().isoformat()`) while
        `start`/`end` here are always UTC-aware (routers/delivery_insights.py's
        `_parse()`), so a raw TEXT range comparison in SQLite would compare
        those two ISO formats inconsistently. `parse_iso` normalizes both to
        aware datetimes before comparing, same as _captured_runs.py does."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, change_type, project_name, component_name, environment_name, "
                "outcome, started_at, finished_at, steps_json FROM deployment_events"
            ).fetchall()
        events = [
            DeploymentEventRecord(
                name=row[0],
                changeType=row[1],
                projectName=row[2],
                componentName=row[3],
                environmentName=row[4],
                outcome=row[5],
                startedAt=row[6],
                finishedAt=row[7],
                steps=json.loads(row[8]) if row[8] else None,
            )
            for row in rows
        ]
        in_window = [e for e in events if start <= parse_iso(e["startedAt"]) < end]
        in_window.sort(key=lambda e: e["startedAt"], reverse=sort_order == "desc")
        return in_window[:limit]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
