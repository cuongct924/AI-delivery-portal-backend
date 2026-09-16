"""SQLite-backed IPredictionLogAdapter — same "local file, no server to
run" precedent as adapters/version_registry_adapter.py's
JsonFileVersionRegistryAdapter, just a real table instead of one JSON
blob (this data is append-only and queried by model_name, which a flat
JSON file handles far worse as it grows).
"""

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from adapters.interfaces import IPredictionLogAdapter, PredictionLogEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions (model_name, id DESC);
"""


class SqlitePredictionLogAdapter(IPredictionLogAdapter):
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("PREDICTION_LOG_PATH", ".state/predictions.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        input_payload: dict[str, object],
        output_payload: dict[str, object] | None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(model_name, model_version, logged_at, input_json, output_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    model_name,
                    model_version,
                    datetime.now(UTC).isoformat(),
                    json.dumps(input_payload),
                    json.dumps(output_payload) if output_payload is not None else None,
                ),
            )

    def list_predictions(self, model_name: str, limit: int = 50) -> list[PredictionLogEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, model_name, model_version, logged_at, input_json, output_json "
                "FROM predictions WHERE model_name = ? ORDER BY id DESC LIMIT ?",
                (model_name, limit),
            ).fetchall()
        return [
            {
                "id": row[0],
                "model_name": row[1],
                "model_version": row[2],
                "logged_at": row[3],
                "input": json.loads(row[4]),
                "output": json.loads(row[5]) if row[5] is not None else None,
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
