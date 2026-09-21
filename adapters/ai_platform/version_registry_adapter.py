"""File-backed IVersionRegistryAdapter for LLMOps prompt/RAG-index version
tracking — not MlflowAdapter, since a prompt isn't an MLflow model artifact.
See docs/llmops-lifecycle-plan.md mục 8 Q2.

In production only ever called with kind="rag-index" (routers/rag.py) —
kind="prompt" moved to MlflowPromptRegistryAdapter (see factory.py's
get_prompt_registry_adapter). Stays genuinely generic over `kind` rather than
guarding it to one value: it was designed and is still tested as a
multi-kind store (tests/test_version_registry_adapter.py exercises kind
independence directly), so a single-kind guard would fight its own design
for a call site that already only ever passes "rag-index" by convention.
"""

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from adapters.ai_platform.interfaces import IVersionRegistryAdapter


class _NameEntry(TypedDict):
    versions: dict[str, dict[str, object]]
    active_version: str | None


# {kind: {name: _NameEntry}} — e.g. {"prompt": {"mlops": {"versions": ...}}}
type _RegistryData = dict[str, dict[str, _NameEntry]]


class JsonFileVersionRegistryAdapter(IVersionRegistryAdapter):
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("LLMOPS_REGISTRY_PATH", ".state/llmops-registry.json"))
        self._lock = threading.Lock()

    def register_version(self, kind: str, name: str, metadata: Mapping[str, object]) -> str:
        with self._lock:
            data = self._read()
            entry = data.setdefault(kind, {}).setdefault(
                name, {"versions": {}, "active_version": None}
            )
            # Same incrementing-int-string shape MLflow uses for versions.
            version = str(len(entry["versions"]) + 1)
            entry["versions"][version] = dict(metadata)
            self._write(data)
            return version

    def list_names(self, kind: str) -> list[str]:
        with self._lock:
            data = self._read()
            return list(data.get(kind, {}))

    def get_version(self, kind: str, name: str, version: str) -> dict[str, object]:
        with self._lock:
            data = self._read()
            try:
                return data[kind][name]["versions"][version]
            except KeyError as exc:
                raise ValueError(f"{kind}/{name} has no version {version!r}") from exc

    def list_versions(self, kind: str, name: str) -> dict[str, dict[str, object]]:
        with self._lock:
            data = self._read()
            return data.get(kind, {}).get(name, {"versions": {}, "active_version": None})[
                "versions"
            ]

    def get_active_version(self, kind: str, name: str) -> str | None:
        with self._lock:
            data = self._read()
            return data.get(kind, {}).get(name, {"versions": {}, "active_version": None})[
                "active_version"
            ]

    def set_active_version(self, kind: str, name: str, version: str) -> None:
        with self._lock:
            data = self._read()
            try:
                entry = data[kind][name]
            except KeyError as exc:
                raise ValueError(f"{kind}/{name} has no registered versions") from exc
            if version not in entry["versions"]:
                raise ValueError(f"{kind}/{name} has no version {version!r}")
            entry["active_version"] = version
            self._write(data)

    def _read(self) -> _RegistryData:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def _write(self, data: _RegistryData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
