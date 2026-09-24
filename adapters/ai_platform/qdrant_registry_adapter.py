"""Qdrant-backed IVersionRegistryAdapter — stores LLMOps version metadata and
per-environment active pointers as points in a dedicated Qdrant collection,
so a RAG index's version lives in the same store as its embeddings (no
separate JSON file to drift from it).

Backs kind="rag-index" (routers/rag.py) and kind="eval-set"
(routers/eval_sets.py). kind="prompt" stays on MLflow's native Prompt
Registry (MlflowPromptRegistryAdapter) — a prompt is text with a first-class
MLflow home, unlike a RAG index pointer or an eval-set.

Two point shapes in the registry collection, distinguished by `record`:
  - "version": one point per (kind, name, version), payload carries the
    caller's metadata (e.g. chunks_ingested/source_paths, or questions).
  - "active": one point per (kind, name, environment), payload carries the
    active version. Kept separate from the version point so activation is a
    single upsert and never rewrites version metadata.

Point ids are deterministic (uuid5 of the natural key), so re-registering or
re-activating the same key overwrites in place instead of duplicating.
"""

import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Record,
    VectorParams,
)

from adapters.ai_platform.interfaces import DEFAULT_ENVIRONMENT, IVersionRegistryAdapter

# Fixed namespace so ids are stable across processes/runs.
_NAMESPACE = uuid.UUID("6f1d2c3a-9b4e-4c7a-8f21-2d5e6a7b8c90")
# Registry points aren't searched by similarity — a single dummy dimension is
# enough; Qdrant just requires a vector config on the collection.
_DUMMY_VECTOR = [0.0]
_RECORD_VERSION = "version"
_RECORD_ACTIVE = "active"
_SCROLL_LIMIT = 1000


class QdrantVersionRegistryAdapter(IVersionRegistryAdapter):
    def __init__(self, url: str | None = None, collection: str | None = None):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection or os.getenv("QDRANT_REGISTRY_COLLECTION", "_llmops_registry")
        self.client = QdrantClient(url=self.url)

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    @staticmethod
    def _version_id(kind: str, name: str, version: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"version:{kind}:{name}:{version}"))

    @staticmethod
    def _active_id(kind: str, name: str, environment: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"active:{kind}:{name}:{environment}"))

    def _scroll(self, conditions: list[Condition]) -> list[Record]:
        self._ensure_collection()
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=conditions),
            limit=_SCROLL_LIMIT,
            with_payload=True,
        )
        return points

    @staticmethod
    def _match(key: str, value: str) -> FieldCondition:
        return FieldCondition(key=key, match=MatchValue(value=value))

    def register_version(self, kind: str, name: str, metadata: Mapping[str, object]) -> str:
        # Same incrementing-int-string shape MLflow uses for versions.
        version = str(len(self.list_versions(kind, name)) + 1)
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._version_id(kind, name, version),
                    vector=_DUMMY_VECTOR,
                    payload={
                        "record": _RECORD_VERSION,
                        "kind": kind,
                        "name": name,
                        "version": version,
                        "metadata": dict(metadata),
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
            ],
        )
        return version

    def list_names(self, kind: str) -> list[str]:
        points = self._scroll([self._match("record", _RECORD_VERSION), self._match("kind", kind)])
        return sorted({str(p.payload["name"]) for p in points if p.payload})

    def get_version(self, kind: str, name: str, version: str) -> dict[str, object]:
        points = self._scroll(
            [
                self._match("record", _RECORD_VERSION),
                self._match("kind", kind),
                self._match("name", name),
                self._match("version", version),
            ]
        )
        if not points or not points[0].payload:
            raise ValueError(f"{kind}/{name} has no version {version!r}")
        return dict(points[0].payload["metadata"])

    def list_versions(self, kind: str, name: str) -> dict[str, dict[str, object]]:
        points = self._scroll(
            [
                self._match("record", _RECORD_VERSION),
                self._match("kind", kind),
                self._match("name", name),
            ]
        )
        return {str(p.payload["version"]): dict(p.payload["metadata"]) for p in points if p.payload}

    def get_active_version(
        self, kind: str, name: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> str | None:
        points = self._scroll(
            [
                self._match("record", _RECORD_ACTIVE),
                self._match("kind", kind),
                self._match("name", name),
                self._match("environment", environment),
            ]
        )
        if not points or not points[0].payload:
            return None
        return str(points[0].payload["version"])

    def set_active_version(
        self, kind: str, name: str, version: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> None:
        if version not in self.list_versions(kind, name):
            raise ValueError(f"{kind}/{name} has no version {version!r}")
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=self._active_id(kind, name, environment),
                    vector=_DUMMY_VECTOR,
                    payload={
                        "record": _RECORD_ACTIVE,
                        "kind": kind,
                        "name": name,
                        "environment": environment,
                        "version": version,
                    },
                )
            ],
        )
