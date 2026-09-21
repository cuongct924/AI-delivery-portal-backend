"""Adapter for the Feature Store (Feast) — self-hosted, offline store for
training-time feature retrieval and online store for inference-time lookups.

Note: needs infra/feature-store/ (a Feast repo — feature_store.yaml plus
entity/feature definitions) provisioned before this can connect for real,
same as kserve_adapter.py needing a real kubeconfig — infra phase item.

Online lookups go over HTTP to `feast serve` (infra/feature-store/Dockerfile,
deployed on worker2's ai-platform-zone — see scripts/setup-3node-infra.sh)
when FEAST_SERVING_URL is set — a genuine low-latency serving path, not just
this process reading the registry file. Offline features and
list_available_features still use the in-process SDK: Feast's HTTP server
only serves online features, and this adapter's own registry copy (baked
into orchestration-api's image, same as feast-serve's) already answers those
without a network hop.
"""

import os
from datetime import datetime

import httpx
import pandas as pd
from feast import FeatureStore

from adapters.ai_platform.interfaces import IFeatureStoreAdapter


class FeastAdapter(IFeatureStoreAdapter):
    def __init__(self, repo_path: str | None = None):
        self.repo_path = repo_path or os.getenv("FEAST_REPO_PATH", "infra/feature-store")
        self.store = FeatureStore(repo_path=self.repo_path)
        self.serving_url = os.getenv("FEAST_SERVING_URL") or None

    def get_offline_features(
        self, entity_ids: list[str], feature_names: list[str], dataset_version: str | None = None
    ) -> list[dict[str, object]]:
        entity_df = pd.DataFrame(
            {"entity_id": entity_ids, "event_timestamp": [datetime.now()] * len(entity_ids)}
        )
        df = self.store.get_historical_features(entity_df=entity_df, features=feature_names).to_df()
        return df.to_dict(orient="records")

    def get_online_features(self, entity_id: str, feature_names: list[str]) -> dict[str, object]:
        if self.serving_url:
            return self._get_online_features_remote(entity_id, feature_names)
        response = self.store.get_online_features(
            features=feature_names, entity_rows=[{"entity_id": entity_id}]
        ).to_dict()
        return {k: v[0] for k, v in response.items()}

    def _get_online_features_remote(
        self, entity_id: str, feature_names: list[str]
    ) -> dict[str, object]:
        # Parallel results per feature, in request order, each a single-value list.
        response = httpx.post(
            f"{self.serving_url}/get-online-features",
            json={"features": feature_names, "entities": {"entity_id": [entity_id]}},
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json()["results"]
        return {
            name: result["values"][0] for name, result in zip(feature_names, results, strict=True)
        }

    def list_available_features(self) -> list[str]:
        return [
            f"{view.name}:{feature.name}"
            for view in self.store.list_feature_views()
            for feature in view.features
        ]
