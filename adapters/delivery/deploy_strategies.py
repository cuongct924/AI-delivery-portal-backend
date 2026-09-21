"""Concrete IDeployTrafficStrategy / IReleaseStrategy implementations for
Golden Path #2 (and the LLM self-hosted-serving golden path). Kept in one
file — 4 small, tightly related classes, not worth 4 separate files.
"""

from adapters.ai_platform.interfaces import IModelRegistryAdapter
from adapters.delivery.interfaces import (
    IDeployTrafficStrategy,
    IInferenceAdapter,
    IReleaseStrategy,
    ReleaseResult,
    TrafficFields,
)


class DirectStrategy(IDeployTrafficStrategy):
    """100% immediately. This cluster runs KServe in RawDeployment/raw mode
    (no Knative/Istio — see infra/ai-platform-zone/kserve/
    test-inferenceservice-cpu.yaml and OpenChoreoInferenceAdapter's own
    raw-mode-label-limit comments), so zero-downtime here comes from
    KServe's own readiness-gated rollout, not Knative revision
    traffic-splitting."""

    def render(self) -> TrafficFields:
        return {}


class BlueGreenStrategy(IDeployTrafficStrategy):
    """Full cutover to a new version, after verifying a prior deploy exists.
    Fixed 100% — no partial split is expressible: OpenChoreo's
    ClusterComponentType has no canary slot in environmentConfigs at all,
    and the API was narrowed to match rather than leave one golden path able
    to express something the other structurally can't."""

    def render(self) -> TrafficFields:
        return {"canaryTrafficPercent": 100}


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
        del manifest_content  # unused — the inference adapter renders its own body
        # Resolves the models:/ shorthand to a real URI; keeps it as-is with no registry.
        storage_uri = (
            self.model_registry_adapter.get_model_artifact_uri(model_name, model_version)
            if self.model_registry_adapter is not None
            else f"models:/{model_name}/{model_version}"
        )
        self.inference_adapter.deploy_model(
            model_name, model_version, storage_uri, traffic_fields=self.traffic_fields
        )
        return {"deployed": True}
