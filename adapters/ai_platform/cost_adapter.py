"""File-backed ICostAdapter — the append-only cost ledger behind the Cost
Insights page.

Every golden-path step that spends money records one entry here (train, RAG
ingest, eval judge, deploy, serve), attributed to the artifact and lifecycle
stage it belongs to. The dashboard then aggregates by stage/dimension instead
of by K8s topology, which is what makes "what did this model version cost to
produce" answerable at all.

File-backed for the same reason the version registry used to be: a demo
needs the ledger to survive a restart without standing up a database, and the
write volume is tiny (one entry per template run, not per request). Swapping
to Postgres later is one new class implementing ICostAdapter.
"""

import json
import os
import threading
from pathlib import Path
from typing import cast

from adapters.ai_platform.interfaces import CostLedgerEntry, ICostAdapter


class JsonFileCostLedgerAdapter(ICostAdapter):
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("COST_LEDGER_PATH", ".state/cost-ledger.json"))
        self._lock = threading.Lock()

    def record_cost(self, entry: CostLedgerEntry) -> None:
        with self._lock:
            entries = self._read()
            entries.append(cast(CostLedgerEntry, dict(entry)))
            self._write(entries)

    def query_costs(
        self,
        start_time: str,
        end_time: str,
        *,
        stage: str | None = None,
        artifact_kind: str | None = None,
        team: str | None = None,
        business_domain: str | None = None,
        environment: str | None = None,
        namespace: str | None = None,
    ) -> list[CostLedgerEntry]:
        with self._lock:
            entries = self._read()
        # ISO 8601 strings compare lexicographically, so no parsing is needed.
        return [
            entry
            for entry in entries
            if start_time <= entry["timestamp"] <= end_time
            and (stage is None or entry["stage"] == stage)
            and (artifact_kind is None or entry["artifact_kind"] == artifact_kind)
            and (team is None or entry["team"] == team)
            and (business_domain is None or entry["business_domain"] == business_domain)
            and (environment is None or entry["environment"] == environment)
            and (namespace is None or entry.get("namespace") == namespace)
        ]

    def _read(self) -> list[CostLedgerEntry]:
        if not self.path.exists():
            return []
        return cast(list[CostLedgerEntry], json.loads(self.path.read_text()))

    def _write(self, entries: list[CostLedgerEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2))
