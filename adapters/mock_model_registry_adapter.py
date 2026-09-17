"""Mock adapter for IModelRegistryAdapter — in-memory stand-in for MLflow.
Returns metrics that pass every evaluations/evaluate_gate.py threshold, whatever
task_type is tagged.
"""

import uuid

import pandas as pd

from adapters.interfaces import (
    DatasetLineageEntry,
    IModelRegistryAdapter,
    ModelRegistration,
    ModelSummary,
    ModelVersionDetails,
)

# Covers every metric in evaluations/evaluate_gate.py's TASK_TYPE_THRESHOLDS,
# so the Evaluate Gate demo passes regardless of task_type.
_FAKE_METRICS: dict[str, float] = {
    "accuracy": 0.92,
    "precision": 0.88,
    "recall": 0.85,
    "f1": 0.86,
    "r2": 0.82,
    "mean_absolute_percentage_error": 0.12,
    "silhouette_score": 0.42,
    "recall_at_k": 0.35,
    "ndcg_at_k": 0.55,
    "anomaly_rate": 0.05,
}


class MockModelRegistryAdapter(IModelRegistryAdapter):
    def __init__(self) -> None:
        self._versions: dict[str, dict[str, ModelVersionDetails]] = {}

    def register_model(
        self, name: str, artifact_uri: str, dataset_version: str | None = None
    ) -> ModelRegistration:
        del artifact_uri  # unused — no real artifact store behind this mock
        versions = self._versions.setdefault(name, {})
        version = str(len(versions) + 1)
        details: ModelVersionDetails = {
            "version": version,
            "run_id": f"mock-run-{uuid.uuid4().hex[:8]}",
            "tags": {},
            "metrics": dict(_FAKE_METRICS),
            "status": "READY",
        }
        if dataset_version is not None:
            details["tags"]["dataset_version"] = dataset_version
        versions[version] = details
        return {"name": name, "version": version}

    def list_models(self, project: str | None = None) -> list[ModelSummary]:
        del project
        return [{"name": name} for name in self._versions]

    def get_model_metrics(self, name: str, version: str) -> dict[str, float]:
        return dict(self._get_details(name, version)["metrics"])

    def get_dataset_lineage(self, name: str, version: str) -> list[DatasetLineageEntry]:
        details = self._get_details(name, version)
        digest = details["tags"].get("dataset_version", "unknown")
        return [
            {
                "name": f"{name}-training-data",
                "digest": digest,
                "source": f"mock://datasets/{name}",
            }
        ]

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self._get_details(name, version)["tags"][key] = value

    def get_model_version_details(self, name: str, version: str) -> ModelVersionDetails:
        return self._get_details(name, version)

    def get_latest_version(self, name: str) -> str:
        # Convenience method, not part of IModelRegistryAdapter — mirrors
        # MlflowAdapter so factory.py can hand either one to routers/models.py.
        versions = self._versions.get(name)
        if not versions:
            raise ValueError(f"Model {name} has no registered versions")
        return max(versions, key=int)

    def list_model_versions(self, name: str) -> list[str]:
        # Mirrors MlflowAdapter.list_model_versions() — same convention.
        return sorted(self._versions.get(name, {}), key=int, reverse=True)

    def get_model_artifact_uri(self, name: str, version: str) -> str:
        # Mirrors MlflowAdapter.get_model_artifact_uri() — no real artifact
        # store behind this mock, so this is just the canonical MLflow
        # shorthand; MockInferenceAdapter never actually fetches it.
        self._get_details(name, version)  # raises if not registered
        return f"models:/{name}/{version}"

    def _get_details(self, name: str, version: str) -> ModelVersionDetails:
        versions = self._versions.get(name)
        if versions is None or version not in versions:
            raise ValueError(f"Model version {name}:{version} is not registered")
        return versions[version]

    def search_runs(
        self, filter_string: str, order_by: list[str] | None = None, max_results: int = 100
    ) -> pd.DataFrame:
        # Mock implementation: return empty DataFrame with expected columns.
        return pd.DataFrame(
            {
                "start_time": pd.Series(dtype="datetime64[ns]"),
                "tags": pd.Series(dtype="object"),
            }
        )
