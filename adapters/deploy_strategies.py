"""Concrete IDeployTrafficStrategy / IReleaseStrategy implementations for
Golden Path #2. Kept in one file — 4 small, tightly related classes, not
worth 4 separate files.
"""

from dataclasses import dataclass

from adapters.interfaces import (
    IDeployTrafficStrategy,
    IInferenceAdapter,
    IModelRegistryAdapter,
    IReleaseStrategy,
    ReleaseResult,
    TrafficFields,
)


class DirectStrategy(IDeployTrafficStrategy):
    """100% immediately — Knative Serving already avoids downtime via
    readiness-gated pod replacement, no extra field needed."""

    def render(self) -> TrafficFields:
        return {}


@dataclass(frozen=True)
class TrafficSplitStrategy(IDeployTrafficStrategy):
    """Canary/A-B/Blue-Green — same `canaryTrafficPercent` mechanism, the
    presets only differ in which percent the Dev-facing form suggests by
    default."""

    percent: int

    def render(self) -> TrafficFields:
        return {"canaryTrafficPercent": self.percent}


class PRGatedStrategy(IReleaseStrategy):
    """Existing behavior, unchanged — a no-op. The caller (routers/models.py)
    still returns manifest_content for the Scaffolder Action to publish as
    a PR, same as before this strategy existed."""

    def release(self, model_name: str, model_version: str, manifest_content: str) -> ReleaseResult:
        del model_name, model_version, manifest_content
        return {"deployed": False}


class InstantStrategy(IReleaseStrategy):
    """Calls the inference adapter directly — no Git/PR."""

    def __init__(
        self,
        inference_adapter: IInferenceAdapter,
        traffic_fields: TrafficFields | None = None,
        model_registry_adapter: IModelRegistryAdapter | None = None,
    ):
        self.inference_adapter = inference_adapter
        self.traffic_fields: TrafficFields = traffic_fields or {}
        self.model_registry_adapter = model_registry_adapter

    def release(self, model_name: str, model_version: str, manifest_content: str) -> ReleaseResult:
        del manifest_content  # unused — KServeAdapter renders its own body
        # Resolves the "models:/<name>/<version>" shorthand to a real artifact
        # location (e.g. "s3://...") when a registry adapter is given — KServe's
        # storage-initializer can't read the shorthand (see its module docstring
        # in openchoreo_inference_adapter.py). Falls back to the shorthand when
        # none is given, matching pre-resolution behavior.
        storage_uri = (
            self.model_registry_adapter.get_model_artifact_uri(model_name, model_version)
            if self.model_registry_adapter is not None
            else f"models:/{model_name}/{model_version}"
        )
        self.inference_adapter.deploy_model(
            model_name, model_version, storage_uri, traffic_fields=self.traffic_fields
        )
        return {"deployed": True}
