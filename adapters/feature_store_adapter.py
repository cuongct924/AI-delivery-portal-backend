"""Adapter for the Feature Store (Feast) — self-hosted, offline store for
training-time feature retrieval and online store for inference-time lookups.

Note: needs infra/feature-store/ (a Feast repo — feature_store.yaml plus
entity/feature definitions) provisioned before this can connect for real,
same as kserve_adapter.py needing a real kubeconfig — infra phase item.
"""

import os
from datetime import datetime

import pandas as pd
from feast import FeatureStore

from adapters.interfaces import IFeatureStoreAdapter


class FeastAdapter(IFeatureStoreAdapter):
    def __init__(self, repo_path: str | None = None):
        self.repo_path = repo_path or os.getenv("FEAST_REPO_PATH", "infra/feature-store")
        self.store = FeatureStore(repo_path=self.repo_path)

    def get_offline_features(
        self, entity_ids: list[str], feature_names: list[str], dataset_version: str | None = None
    ) -> list[dict[str, object]]:
        entity_df = pd.DataFrame(
            {"entity_id": entity_ids, "event_timestamp": [datetime.now()] * len(entity_ids)}
        )
        df = self.store.get_historical_features(entity_df=entity_df, features=feature_names).to_df()
        return df.to_dict(orient="records")

    def get_online_features(self, entity_id: str, feature_names: list[str]) -> dict[str, object]:
        response = self.store.get_online_features(
            features=feature_names, entity_rows=[{"entity_id": entity_id}]
        ).to_dict()
        return {k: v[0] for k, v in response.items()}

    def list_available_features(self) -> list[str]:
        return [
            f"{view.name}:{feature.name}"
            for view in self.store.list_feature_views()
            for feature in view.features
        ]
