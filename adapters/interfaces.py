"""Shared contract for every Adapter — implemented via the Adapter Pattern.

Principle: switching from Mock/MLflow to a real self-hosted backend only
requires adding one new class that implements this interface, without
touching code that already depends on the interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict


class ModelRegistration(TypedDict):
    name: str
    version: str


class ModelSummary(TypedDict):
    name: str


class DatasetLineageEntry(TypedDict):
    name: str
    digest: str
    source: str


class ModelVersionDetails(TypedDict):
    version: str
    run_id: str
    tags: dict[str, str]
    metrics: dict[str, float]
    status: str


class IModelRegistryAdapter(ABC):
    @abstractmethod
    def register_model(self, name: str, artifact_uri: str) -> ModelRegistration: ...

    @abstractmethod
    def list_models(self, project: str | None = None) -> list[ModelSummary]: ...

    @abstractmethod
    def get_model_metrics(self, name: str, version: str) -> dict[str, float]: ...

    @abstractmethod
    def get_dataset_lineage(self, name: str, version: str) -> list[DatasetLineageEntry]: ...

    @abstractmethod
    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None: ...

    @abstractmethod
    def get_model_version_details(self, name: str, version: str) -> ModelVersionDetails: ...


class IVersionRegistryAdapter(ABC):
    """Tracks versions of an artifact that isn't a trained model (prompt
    text, RAG index pointer) and which one is currently active — the
    LLMOps equivalent of IModelRegistryAdapter's "model version", without
    assuming an MLflow-loadable artifact exists.

    `metadata`/return shape is intentionally `dict[str, object]`, not a
    TypedDict — it's polymorphic per `kind` ("prompt" vs "rag-index" carry
    different fields), so no single fixed key set exists to declare.
    """

    @abstractmethod
    def register_version(self, kind: str, name: str, metadata: Mapping[str, object]) -> str: ...

    @abstractmethod
    def list_names(self, kind: str) -> list[str]: ...

    @abstractmethod
    def get_version(self, kind: str, name: str, version: str) -> dict[str, object]: ...

    @abstractmethod
    def list_versions(self, kind: str, name: str) -> dict[str, dict[str, object]]: ...

    @abstractmethod
    def get_active_version(self, kind: str, name: str) -> str | None: ...

    @abstractmethod
    def set_active_version(self, kind: str, name: str, version: str) -> None: ...


class IInferenceAdapter(ABC):
    """`deploy_model`/`get_inference_status` return `dict[str, object]`, not
    a TypedDict — they pass through the Kubernetes/KServe API's raw
    InferenceService resource, whose shape is large and versioned by
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
    def predict(self, name: str, payload: dict[str, object]) -> dict[str, object]: ...


class TrafficFields(TypedDict, total=False):
    canaryTrafficPercent: int


class IDeployTrafficStrategy(ABC):
    """How traffic moves to the new model version — Golden Path #2.

    Direct and TrafficSplit (Canary/A-B/Blue-Green) are the only 2
    concrete strategies: KServe's `canaryTrafficPercent` field is one
    mechanism that Canary/A-B/Blue-Green only differ in *intent* over —
    not 3 separate classes.
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


class WorkflowStatus(TypedDict):
    name: str
    phase: str | None
    message: str | None


class IWorkflowAdapter(ABC):
    """`trigger_workflow` returns `dict[str, object]`, not a TypedDict — it
    passes through the Argo Server API's raw workflow resource."""

    @abstractmethod
    def trigger_workflow(
        self, template_name: str, parameters: dict[str, str]
    ) -> dict[str, object]: ...

    @abstractmethod
    def get_workflow_status(self, workflow_name: str) -> WorkflowStatus: ...


class UpsertResult(TypedDict):
    status: str


class SearchHit(TypedDict):
    id: str
    score: float
    payload: dict[str, object]


class IVectorStoreAdapter(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: Sequence[Mapping[str, object]],
        collection: str | None = None,
    ) -> UpsertResult: ...

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int = 5, collection: str | None = None
    ) -> list[SearchHit]: ...


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


class ChatCompletionMessage(TypedDict):
    role: str
    content: str | None
    tool_calls: NotRequired[list[ToolCall]]


class ChatCompletionChoice(TypedDict):
    message: ChatCompletionMessage
    finish_reason: str | None


class ChatCompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(TypedDict):
    id: str
    model: str
    choices: list[ChatCompletionChoice]
    usage: NotRequired[ChatCompletionUsage]
    # Injected by LiteLLMGatewayAdapter from a response header — None when
    # the model has no cost entry configured in litellm-config.yaml.
    response_cost_usd: NotRequired[float | None]


class ILLMGatewayAdapter(ABC):
    @abstractmethod
    def chat_completion(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> ChatCompletionResponse: ...

    @abstractmethod
    def list_models(self) -> list[dict[str, object]]: ...

    @abstractmethod
    def embed(self, model: str, input_texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def get_spend_report(
        self, start_date: str, end_date: str, group_by: str | None = None
    ) -> list[dict[str, object]]:
        """Aggregate spend over [start_date, end_date] (YYYY-MM-DD). Shape
        varies by `group_by` ("team"/"customer"/None for api_key level) —
        same polymorphic-dict reasoning as IVersionRegistryAdapter above,
        no single fixed key set exists across groupings."""
        ...


class IFeatureStoreAdapter(ABC):
    """Feature columns are named by the caller's `feature_names` — the
    returned rows are genuinely dynamic-key, not a fixed TypedDict shape."""

    @abstractmethod
    def get_offline_features(
        self, entity_ids: list[str], feature_names: list[str], dataset_version: str | None = None
    ) -> list[dict[str, object]]: ...

    @abstractmethod
    def get_online_features(
        self, entity_id: str, feature_names: list[str]
    ) -> dict[str, object]: ...


class NotebookStatus(TypedDict):
    notebook_id: str
    url: str | None
    active: bool


class NotebookDeletion(TypedDict):
    notebook_id: str
    deleted: bool


class INotebookAdapter(ABC):
    @abstractmethod
    def create_notebook(
        self, environment: str, ram_gb: int, gpu_type: str | None = None
    ) -> NotebookStatus: ...

    @abstractmethod
    def get_notebook_status(self, notebook_id: str) -> NotebookStatus: ...

    @abstractmethod
    def delete_notebook(self, notebook_id: str) -> NotebookDeletion: ...


class DatasetInfo(TypedDict):
    name: str
    uri: str
    size_bytes: int
    source: str


class IObjectStorageAdapter(ABC):
    """Discovers datasets already available for training — backs the
    Scaffolder UI's dataset picker so a user chooses from what's actually
    there instead of typing a `file://` path blind.

    Two implementations, merged by `CompositeObjectStorageAdapter` into
    one listing: `MinioObjectStorageAdapter` (MinIO/S3, `source="s3"`) and
    `LocalFileObjectStorageAdapter` (the `data/` working-tree checkout, for
    local dev without MinIO running, `source="local"`).

    `DatasetInfo.uri` is always a `file://` path the training pod can
    actually open (it reads from a hostPath mount, not the bucket
    directly) — the object store is only consulted here for what dataset
    *names* exist.
    """

    @abstractmethod
    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]: ...
