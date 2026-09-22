"""Shared contract for every delivery/ Adapter (deploy/promote/workflow —
OpenChoreo-facing) — implemented via the Adapter Pattern.

Principle: switching from Mock to a real backend only requires adding one
new class that implements this interface, without touching code that
already depends on the interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, TypedDict


class PromotionStatus(TypedDict):
    project: str
    component: str
    # env name -> currently bound projectRelease, None if never promoted here.
    environments: dict[str, str | None]
    # True when staging has a release production doesn't — promotable now.
    prod_pending_approval: bool


class IPromotionAdapter(ABC):
    """OpenChoreo DeploymentPipeline/ProjectReleaseBinding-based staging/prod
    promotion — replaces the abandoned Kargo-based pipeline. The read-only
    promotion-status MCP tool now lives in OpenChoreo's own Control Plane MCP
    server (`get_release_binding`/`list_deployment_pipelines`), not in
    agents/mcp-servers/.

    Staging/production promotion is PR-gated, same as dev's
    PRGatedStrategy: `resolve_promotion_release()`/`resolve_rollback_release()`
    only read state (a router renders their result as a ProjectReleaseBinding
    manifest for a human to review as a PR); `confirm_promotion()` is the one
    method that actually writes to the cluster, called once that PR is
    merged. Never an agent/MCP tool — only reachable via a Scaffolder Golden
    Path the Dev runs.
    """

    @abstractmethod
    def get_promotion_status(self) -> PromotionStatus: ...

    @abstractmethod
    def resolve_promotion_release(self, target_environment: str) -> str:
        """Returns whatever release is currently bound in the pipeline's
        source environment for `target_environment` — what a promotion
        would move forward, without writing anything. Raises ValueError if
        `target_environment` isn't a valid promotion target for wherever
        the project currently is (e.g. promoting to "production" before
        anything has reached "staging")."""
        ...

    @abstractmethod
    def resolve_rollback_release(self, environment: str) -> str:
        """Returns whatever release `environment` would roll back to right
        now — the one bound there immediately before its last
        promotion/rollback, without writing anything. Raises ValueError
        when there's nothing bound in `environment` yet, or nothing
        recorded to roll back to (its current binding was never promoted
        over)."""
        ...

    @abstractmethod
    def confirm_promotion(self, environment: str, project_release: str) -> PromotionStatus:
        """Writes `project_release` as `environment`'s binding — the real
        OpenChoreo/cluster mutation for both promote and rollback (they
        differ only in which release a router resolved beforehand). Called
        once a human has reviewed the manifest a router rendered from
        `resolve_promotion_release()`/`resolve_rollback_release()` — e.g.
        merged the PR it was published as."""
        ...


class DeployStatus(TypedDict):
    """Adapter-owned deploy status — normalizes away the raw KServe/OpenChoreo
    resource shape so callers (routers/models.py, routers/llm_serving.py)
    never traverse spec/metadata/status dicts themselves. `pr_url` is
    deliberately excluded — that's an MLflow model-version tag the router
    cross-references separately, not inference-adapter data."""

    deployed: bool
    ready: bool
    live_version: str | None
    traffic_percent: int | None


class IInferenceAdapter(ABC):
    """Standard model-serving golden path (routers/models.py) —
    OpenChoreo-backed. `deploy_model`/`get_inference_status` return
    `dict[str, object]`, not a TypedDict — they pass through the raw
    Kubernetes/OpenChoreo resource, whose shape is large and versioned by
    Kubernetes itself, not by this codebase."""

    @abstractmethod
    def deploy_model(
        self,
        name: str,
        version: str,
        model_uri: str,
        traffic_fields: Mapping[str, object] | None = None,
    ) -> dict[str, object]: ...

    @abstractmethod
    def get_inference_status(self, name: str) -> dict[str, object]: ...

    @abstractmethod
    def get_deploy_status(self, name: str) -> DeployStatus: ...

    @abstractmethod
    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]: ...


class IGpuInferenceAdapter(ABC):
    """LLM self-hosted-serving golden path (routers/llm_serving.py) —
    KServe-backed. No OpenChoreo ClusterComponentType exists for GPU/vLLM
    serving, so this stays a separate interface/backend from
    `IInferenceAdapter` rather than a third method bolted onto it — see
    adapters/delivery/gpu_inference_adapter.py's module docstring."""

    @abstractmethod
    def deploy_llm_model(
        self,
        name: str,
        version: str,
        huggingface_model_id: str,
        serving_runtime_name: str,
        gpu_count: int,
        vllm_quantization: str | None,
        max_context_length: int,
        traffic_fields: Mapping[str, object] | None = None,
        hf_token_secret_ref: str | None = None,
    ) -> dict[str, object]: ...

    @abstractmethod
    def get_inference_status(self, name: str) -> dict[str, object]: ...

    @abstractmethod
    def get_deploy_status(self, name: str) -> DeployStatus: ...

    @abstractmethod
    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]: ...


class TrafficFields(TypedDict, total=False):
    canaryTrafficPercent: int


class IDeployTrafficStrategy(ABC):
    """How traffic moves to the new model version — Golden Path #2 and the
    LLM self-hosted-serving golden path both use this.

    Direct and BlueGreen are the only 2 concrete strategies — a genuine
    partial traffic split isn't supported: OpenChoreo's ClusterComponentType
    has no canary slot at all, and the API surface was narrowed to match
    rather than keep one golden path able to express something the other
    structurally can't.
    """

    @abstractmethod
    def render(self) -> TrafficFields:
        """Fields to merge into the InferenceService's spec.predictor
        block — {} for Direct, {"canaryTrafficPercent": N} for TrafficSplit."""


class ReleaseResult(TypedDict):
    deployed: bool


class IReleaseStrategy(ABC):
    """How a deploy gets approved — PR-gated (default, unchanged) vs
    Instant (calls the inference adapter directly, no Git/PR)."""

    @abstractmethod
    def release(self, model_name: str, model_version: str, manifest_content: str) -> ReleaseResult:
        """Performs the release action. PRGatedStrategy is a no-op — the
        caller still publishes manifest_content as a PR itself. Instant
        actually deploys and returns {"deployed": True}."""


class WorkflowStepTiming(TypedDict):
    name: str
    phase: str | None
    started_at: str | None
    finished_at: str | None


class WorkflowStatus(TypedDict):
    name: str
    phase: str | None
    message: str | None
    # Populated once workflow starts — backs Lead Time metrics in dora_metrics.py.
    started_at: str | None
    finished_at: str | None
    steps: list[WorkflowStepTiming]


class IWorkflowAdapter(ABC):
    """`trigger_workflow` returns `dict[str, object]`, not a TypedDict — it
    passes through the backend's raw workflow resource (an OpenChoreo
    WorkflowRun, or a mock equivalent)."""

    @abstractmethod
    def trigger_workflow(
        self, template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]: ...

    @abstractmethod
    def get_workflow_status(self, workflow_name: str) -> WorkflowStatus: ...


type DoraGranularity = Literal["daily", "weekly", "monthly"]
type ChangeType = Literal["infra", "model", "rag_index", "prompt"]
type LifecyclePhase = Literal["data_prep", "train", "eval", "deploy"]
type FailureClass = Literal["infra", "semantic"]
type SemanticFailureType = Literal[
    "accuracy_drop", "drift", "hallucination", "guardrail", "prompt_injection"
]
type RecoveryStrategy = Literal["rollback", "fallback", "guardrail", "retrain"]


class DeliveryScope(TypedDict):
    namespace: str
    project: str | None
    component: str | None
    environment: str | None


class DeliveryDeployment(TypedDict):
    """Twin of routers/delivery_insights.py's `DoraDeployment` Pydantic
    model — the ML/LLM fields are all optional-valued so an infra-only row
    keeps today's shape."""

    deployedAt: str
    projectName: str
    componentName: str
    environmentName: str
    componentRelease: str
    commit: str
    outcome: Literal["success", "failed", "in_progress"]
    failedBy: str
    failureReason: str
    incidentId: str
    leadTimeMs: int | None
    changeType: ChangeType | None
    driftTriggered: bool
    evalCoverage: float | None
    leadTimeBreakdown: dict[LifecyclePhase, int] | None
    evalBottleneck: LifecyclePhase | None
    failureClass: FailureClass | None
    semanticType: SemanticFailureType | None
    evalScore: float | None
    baselineScore: float | None
    driftScore: float | None
    recoveryStrategy: RecoveryStrategy | None
    modelVersion: str | None
    promptVersion: str | None
    ragIndexVersion: str | None


class DeliveryFrequencySummary(TypedDict):
    total: int
    perDay: float
    classification: str
    deltaPct: float | None


class DeliveryLeadTimeSummary(TypedDict):
    p50Ms: int | None
    p95Ms: int | None
    coverage: float
    classification: str
    deltaPct: float | None


class DeliveryChangeFailureRateSummary(TypedDict):
    rate: float
    failed: int
    total: int
    classification: str
    deltaPct: float | None
    infraCfr: float | None
    semanticCfr: float | None


class DeliveryMttrSummary(TypedDict):
    meanMs: int | None
    p50Ms: int | None
    recoveries: int
    classification: str
    deltaPct: float | None


class DeliverySummary(TypedDict):
    deploymentFrequency: DeliveryFrequencySummary | None
    leadTime: DeliveryLeadTimeSummary | None
    changeFailureRate: DeliveryChangeFailureRateSummary | None
    mttr: DeliveryMttrSummary | None


class DeliveryFrequencyPoint(TypedDict):
    bucketStart: str
    count: int


class DeliveryLeadTimePoint(TypedDict):
    bucketStart: str
    p50Ms: int
    p75Ms: int
    p95Ms: int


class DeliveryChangeFailureRatePoint(TypedDict):
    bucketStart: str
    rate: float
    failed: int
    total: int


class DeliveryMttrPoint(TypedDict):
    bucketStart: str
    meanMs: int
    p50Ms: int
    count: int


class DeliverySeries(TypedDict):
    deploymentFrequency: list[DeliveryFrequencyPoint] | None
    leadTime: list[DeliveryLeadTimePoint] | None
    changeFailureRate: list[DeliveryChangeFailureRatePoint] | None
    mttr: list[DeliveryMttrPoint] | None


class DeliveryAvailability(TypedDict):
    collecting: bool
    deliveryEvents: bool
    evalPipeline: bool
    driftMonitor: bool
    guardrails: bool


class DeliveryWindow(TypedDict):
    startTime: str
    endTime: str
    generatedAt: str


class DeliveryMetricsResult(TypedDict):
    dataAvailability: DeliveryAvailability
    scope: DeliveryScope
    granularity: DoraGranularity
    window: DeliveryWindow
    summary: DeliverySummary
    series: DeliverySeries


class DeliveryDeploymentsResult(TypedDict):
    deployments: list[DeliveryDeployment]
    totalCount: int
    tookMs: int


class IDeliveryObserverAdapter(ABC):
    """Delivery Insights (DORA) observer for the Portal dashboard — mirrors
    the shape OpenChoreo's real Observer API returns, so
    routers/delivery_insights.py stays a thin pass-through regardless of
    which class backs it (synthetic/captured-JSON mock data vs. real
    Prometheus + MLflow)."""

    @abstractmethod
    def query_metrics(
        self,
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        granularity: DoraGranularity,
        metrics: list[str] | None,
    ) -> DeliveryMetricsResult: ...

    @abstractmethod
    def query_deployments(
        self,
        scope: DeliveryScope,
        start: datetime,
        end: datetime,
        limit: int,
        sort_order: Literal["asc", "desc"],
    ) -> DeliveryDeploymentsResult: ...
