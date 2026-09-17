"""In-memory Mock implementation of IEvalResultAdapter — used when
USE_MOCK_ADAPTERS=true (default for local dev and CI)."""

from datetime import datetime
from typing import Any

from adapters.interfaces import IEvalResultAdapter


class MockEvalResultAdapter(IEvalResultAdapter):
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def log_judge_result(
        self,
        kind: str,
        name: str,
        version: str,
        judge_result: object,
        passed: bool,
    ) -> None:
        self._records.append(
            {
                "kind": kind,
                "name": name,
                "version": version,
                "judge_result": judge_result,
                "passed": passed,
                "timestamp": datetime.now(),
            }
        )

    def get_last_failure_at(self, kind: str, name: str) -> datetime | None:
        failures = [
            r["timestamp"]
            for r in self._records
            if r["kind"] == kind and r["name"] == name and r["passed"] is False
        ]
        if not failures:
            return None
        return max(failures)
