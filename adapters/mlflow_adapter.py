"""Real adapter using the MLflow SDK — connects to the mlflow service in docker-compose.yml."""

import os

import mlflow
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from adapters.interfaces import (
    DatasetLineageEntry,
    IModelRegistryAdapter,
    ModelRegistration,
    ModelSummary,
    ModelVersionDetails,
)


def _to_dataframe(result: list | pd.DataFrame) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    return pd.DataFrame(result)


# TODO: expose via orchestration-api for the `orchestration:register-model`
# Custom Scaffolder Action (Golden Path #1) to call.
class MlflowAdapter(IModelRegistryAdapter):
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model(
        self, name: str, artifact_uri: str, dataset_version: str | None = None
    ) -> ModelRegistration:
        # No `version` param — MLflow assigns it, auto-incrementing per name.
        result = mlflow.register_model(model_uri=artifact_uri, name=name)
        if dataset_version is not None:
            # Caller already resolved the DVC hash; see data/README.md.
            self.client.set_model_version_tag(
                name, result.version, "dataset_version", dataset_version
            )
        return {"name": result.name, "version": result.version}

    def list_models(self, project: str | None = None) -> list[ModelSummary]:
        return [{"name": m.name} for m in self.client.search_registered_models()]

    def get_model_metrics(self, name: str, version: str) -> dict[str, float]:
        mv = self.client.get_model_version(name=name, version=version)
        if mv.run_id is None:
            raise ValueError(f"Model version {name}:{version} has no associated run_id")
        run = self.client.get_run(mv.run_id)
        return dict(run.data.metrics)

    def get_dataset_lineage(self, name: str, version: str) -> list[DatasetLineageEntry]:
        # Reads lineage logged at training time (mlflow.log_input), not a
        # tag set here — a run can log more than one dataset.
        mv = self.client.get_model_version(name=name, version=version)
        if mv.run_id is None:
            raise ValueError(f"Model version {name}:{version} has no associated run_id")
        run = self.client.get_run(mv.run_id)
        return [
            {"name": d.dataset.name, "digest": d.dataset.digest, "source": str(d.dataset.source)}
            for d in run.inputs.dataset_inputs
        ]

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.client.set_model_version_tag(name, version, key, value)

    def get_model_version_details(self, name: str, version: str) -> ModelVersionDetails:
        # Translated to ValueError (a caller can catch without importing
        # mlflow — same reasoning as IObjectStorageAdapter's docstring on
        # keeping the interface swappable) rather than letting MLflow's own
        # RestException surface — a version that was never registered, or
        # one whose name is misspelled, is routine caller input, not a
        # genuine server fault.
        try:
            mv = self.client.get_model_version(name=name, version=version)
        except RestException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise ValueError(f"model version {name}:{version} not found") from e
            raise
        if mv.run_id is None:
            raise ValueError(f"Model version {name}:{version} has no associated run_id")
        run = self.client.get_run(mv.run_id)
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "tags": dict(mv.tags),
            "metrics": dict(run.data.metrics),
            "status": mv.status,
        }

    def get_latest_version(self, name: str) -> str:
        # Convenience method, not part of IModelRegistryAdapter — same
        # precedent as QdrantAdapter.ensure_collection() in vector_db_adapter.py.
        versions = self.client.search_model_versions(f"name='{name}'")
        if not versions:
            raise ValueError(f"Model {name} has no registered versions")
        return max(versions, key=lambda mv: int(mv.version)).version

    def get_model_artifact_uri(self, name: str, version: str) -> str:
        """Resolves "models:/<name>/<version>" (this repo's existing
        storage_uri convention — routers/models.py, adapters/deploy_strategies.py)
        to the real underlying artifact location (e.g. "s3://...").
        Needed because KServe's storage-initializer isn't an MLflow client
        — it can't read the "models:/" shorthand at all (confirmed for
        real: "Cannot recognize storage type for models:/...";
        adapters/openchoreo_inference_adapter.py's module docstring has
        the full story). Not part of IModelRegistryAdapter — same
        precedent as get_latest_version()/list_model_versions() above;
        MockModelRegistryAdapter has no equivalent since KServe/OpenChoreo
        deploys never run against the mock registry.
        """
        return self.client.get_model_version_download_uri(name, version)

    def list_model_versions(self, name: str) -> list[str]:
        # Convenience method, not part of IModelRegistryAdapter — same
        # precedent as get_latest_version above. Powers the Evaluate &
        # Deploy Model template's version dropdown: listing only versions
        # that actually exist removes the free-text typo class entirely,
        # instead of only catching it after the fact (get_model_version_summary's 404).
        versions = self.client.search_model_versions(f"name='{name}'")
        return sorted((mv.version for mv in versions), key=int, reverse=True)

    def search_runs(
        self, filter_string: str, order_by: list[str] | None = None, max_results: int = 100
    ) -> pd.DataFrame:
        result = mlflow.search_runs(
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results,
        )
        return _to_dataframe(result)
