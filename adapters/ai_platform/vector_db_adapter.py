"""Adapter for the Vector DB (Qdrant) — powers the RAG architecture (semantic
search over internal docs/techdocs/runbooks). Deployed via
infra/ai-platform-zone/qdrant.yaml.

Note: this only accepts pre-computed vectors — the embedding step (Voyage AI
or self-hosted) happens at the layer calling this adapter, see
infra/vector-dbs/README.md.

Versioning: every point carries an `index_version` payload field (set by
`upsert`), and `search` can filter to one version. That is what makes a RAG
index version a real retrieval boundary — activating a version changes which
points a query sees, instead of every version's points being mixed in one
collection. The version *catalog* (which versions exist, which is active per
environment) lives in Qdrant too, see qdrant_registry_adapter.py.
"""

import os
from collections.abc import Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from adapters.ai_platform.interfaces import IVectorStoreAdapter, SearchHit, UpsertResult


class QdrantAdapter(IVectorStoreAdapter):
    def __init__(self, url: str | None = None, collection: str = "ai-delivery-portal-docs"):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self, vector_size: int = 1536, collection: str | None = None) -> None:
        collection = collection or self.collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: Sequence[Mapping[str, object]],
        collection: str | None = None,
        index_version: str | None = None,
    ) -> UpsertResult:
        points = []
        for i, v, p in zip(ids, vectors, payloads, strict=True):
            payload = dict(p)
            if index_version is not None:
                payload["index_version"] = index_version
            points.append(PointStruct(id=i, vector=v, payload=payload))
        result = self.client.upsert(collection_name=collection or self.collection, points=points)
        return {"status": str(result.status)}

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        collection: str | None = None,
        index_version: str | None = None,
    ) -> list[SearchHit]:
        query_filter = None
        if index_version is not None:
            query_filter = Filter(
                must=[FieldCondition(key="index_version", match=MatchValue(value=index_version))]
            )
        hits = self.client.query_points(
            collection_name=collection or self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        # IDs are strings throughout this interface, though qdrant-client also allows int/UUID.
        return [SearchHit(id=str(h.id), score=h.score, payload=h.payload or {}) for h in hits]
