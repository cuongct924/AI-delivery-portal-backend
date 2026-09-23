"""File-backed IVersionRegistryAdapter for LLMOps prompt/RAG-index version
tracking — not MlflowAdapter, since a prompt isn't an MLflow model artifact.
See docs/llmops-lifecycle-plan.md mục 8 Q2.

In production called with kind="rag-index" (routers/rag.py) and
kind="eval-set" (routers/eval_sets.py) — kind="prompt" moved to
MlflowPromptRegistryAdapter (see factory.py's get_prompt_registry_adapter).
Stays genuinely generic over `kind` rather than guarding it to one value:
it was designed and is still tested as a multi-kind store
(tests/test_version_registry_adapter.py exercises kind independence
directly), which is exactly what let eval-set reuse it for free.
"""

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from adapters.ai_platform.interfaces import DEFAULT_ENVIRONMENT, IVersionRegistryAdapter


class _NameEntry(TypedDict):
    versions: dict[str, dict[str, object]]
    # {environment: version} — e.g. {"production": "3", "development": "4"}.
    active_versions: dict[str, str]


# {kind: {name: _NameEntry}} — e.g. {"prompt": {"mlops": {"versions": ...}}}
type _RegistryData = dict[str, dict[str, _NameEntry]]


def _empty_entry() -> _NameEntry:
    return {"versions": {}, "active_versions": {}}


def _active_versions(entry: Mapping[str, object]) -> dict[str, str]:
    """Reads an entry's per-environment active versions, transparently
    migrating a pre-environment entry (a bare `"active_version"` string, or
    key absent) into the `{DEFAULT_ENVIRONMENT: version}` shape — a file
    written before this adapter tracked environments at all still reads
    correctly, with its one active version treated as production's."""
    active_versions = entry.get("active_versions")
    if isinstance(active_versions, dict):
        return cast(dict[str, str], active_versions)
    legacy_version = entry.get("active_version")
    return {DEFAULT_ENVIRONMENT: cast(str, legacy_version)} if legacy_version else {}


class JsonFileVersionRegistryAdapter(IVersionRegistryAdapter):
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv("LLMOPS_REGISTRY_PATH", ".state/llmops-registry.json"))
        self._lock = threading.Lock()

    def register_version(self, kind: str, name: str, metadata: Mapping[str, object]) -> str:
        with self._lock:
            data = self._read()
            entry = data.setdefault(kind, {}).setdefault(name, _empty_entry())
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
            return data.get(kind, {}).get(name, _empty_entry())["versions"]

    def get_active_version(
        self, kind: str, name: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> str | None:
        with self._lock:
            data = self._read()
            entry = data.get(kind, {}).get(name, _empty_entry())
            return _active_versions(entry).get(environment)

    def set_active_version(
        self, kind: str, name: str, version: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> None:
        with self._lock:
            data = self._read()
            try:
                entry = data[kind][name]
            except KeyError as exc:
                raise ValueError(f"{kind}/{name} has no registered versions") from exc
            if version not in entry["versions"]:
                raise ValueError(f"{kind}/{name} has no version {version!r}")
            entry.setdefault("active_versions", {})[environment] = version
            self._write(data)

    def _read(self) -> _RegistryData:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def _write(self, data: _RegistryData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
