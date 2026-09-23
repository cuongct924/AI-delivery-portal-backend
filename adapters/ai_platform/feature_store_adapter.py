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
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
from feast import FeatureStore, FeatureView

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
        try:
            df = self.store.get_historical_features(
                entity_df=entity_df, features=feature_names
            ).to_df()
        except TypeError:
            # Feast 0.66 + dask 2026.7 bug: when NO entity id matches the
            # offline store, the point-in-time join is all-NaT and
            # _filter_ttl compares that tz-naive column against the tz-aware
            # entity timestamp, raising "Invalid comparison between
            # dtype=datetime64[ns] and DatetimeArray". A plain left merge is
            # exactly what the join reduces to when every row is unmatched,
            # so fall back to it rather than 500ing the whole training run.
            df = self._offline_features_left_merge(entity_ids, feature_names)
        return df.to_dict(orient="records")

    def _offline_features_left_merge(
        self, entity_ids: list[str], feature_names: list[str]
    ) -> pd.DataFrame:
        """Plain left merge on entity_id — see get_offline_features."""
        result = pd.DataFrame({"entity_id": entity_ids})
        by_view: dict[str, list[str]] = {}
        for name in feature_names:
            view_name, feature_name = name.split(":", 1)
            by_view.setdefault(view_name, []).append(feature_name)
        for view_name, features in by_view.items():
            view = self.store.get_feature_view(view_name)
            data = self._read_feature_source(view)
            data = cast(pd.DataFrame, data[["entity_id", *features]]).drop_duplicates(
                subset=["entity_id"], keep="last"
            )
            result = result.merge(data, on="entity_id", how="left")
        return result

    def _read_feature_source(self, view: FeatureView) -> pd.DataFrame:
        path = getattr(view.batch_source, "path", None)
        if not isinstance(path, str):
            raise ValueError(f"feature view {view.name!r} has no file source path")
        if "://" not in path and not Path(path).is_absolute():
            path = str(Path(self.repo_path) / path)
        return pd.read_parquet(path)

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

    def list_entity_ids(self) -> list[str]:
        """Every entity id the offline store actually holds — lets the UI warn
        when a dataset's chosen entity column shares no values with the store
        (the features would silently come back all-NaN)."""
        ids: set[str] = set()
        for view in self.store.list_feature_views():
            data = self._read_feature_source(view)
            ids.update(data["entity_id"].astype(str))
        return sorted(ids)
