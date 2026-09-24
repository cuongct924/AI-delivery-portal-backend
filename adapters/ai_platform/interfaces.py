"""Shared contract for every ai_platform/ Adapter (model registry/
experiments/feature store/vector store/gateway — Viettel AI Platform-facing)
— implemented via the Adapter Pattern.

Principle: switching from Mock/MLflow to a real self-hosted backend only
requires adding one new class that implements this interface, without
touching code that already depends on the interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, NotRequired, TypedDict


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

    @abstractmethod
    def get_model_artifact_uri(self, name: str, version: str) -> str:
        """Resolves to a URI KServe's storage-initializer can actually read
        (e.g. "s3://...") — the "models:/<name>/<version>" shorthand alone
        isn't recognized by it. See adapters/delivery/deploy_strategies.py's
        InstantStrategy for the caller."""
        ...

    @abstractmethod
    def search_runs(
        self, filter_string: str, order_by: list[str] | None = None, max_results: int = 100
    ) -> object:
        """Searches MLflow runs with the given filter string, returning a
        pandas DataFrame (columns like start_time, tags). Used for finding
        drift-detection runs for MTTR calculation."""
        ...


#: Preserves every existing caller's behavior unchanged — chat.py and the
#: ai-observability MCP tools call get_active_version(kind, name) with no
#: environment, and must keep reading/writing exactly what "activate" meant
#: before environments existed (the one version real traffic uses).
DEFAULT_ENVIRONMENT: Final[str] = "production"


def normalize_version(version: str) -> str:
    """Canonicalize a user-supplied version to the form the registry stores.

    The Scaffolder UI renders versions as "v1" (OptionPickerField's
    formatOption) while every registry stores them as "1" — a user who types
    the displayed form would otherwise miss every lookup (and a bare
    `int(version)` would raise). Strips a single leading "v"/"V" only when
    what follows is all digits, so a genuinely non-numeric version is left
    untouched.
    """
    if len(version) > 1 and version[0] in ("v", "V") and version[1:].isdigit():
        return version[1:]
    return version


class IVersionRegistryAdapter(ABC):
    """Tracks versions of an artifact that isn't a trained model (prompt
    text, RAG index pointer) and which one is currently active *per
    environment* — the LLMOps equivalent of a "model version" plus
    OpenChoreo's per-environment ReleaseBinding, without assuming an
    MLflow-loadable artifact or an OpenChoreo Component exists.

    `metadata`/return shape is `dict[str, object]`, not a TypedDict — it's
    polymorphic per `kind` ("prompt" vs "rag-index" carry different fields).
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
    def get_active_version(
        self, kind: str, name: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> str | None: ...

    @abstractmethod
    def set_active_version(
        self, kind: str, name: str, version: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> None: ...


class PredictionLogEntry(TypedDict):
    id: int
    model_name: str
    model_version: str
    # ISO 8601 — sqlite3 has no native timestamp type; store a plain string.
    logged_at: str
    input: dict[str, object]
    output: dict[str, object] | None


class IPredictionLogAdapter(ABC):
    """Logs predict-time (input, output) pairs — the data for Golden Path #4
    (data drift monitoring), started now rather than waiting (see docs/
    mlops-lifecycle-software-template.md). Not automatic: nothing here proxies
    real predict traffic (the inference adapters' predict() tells callers to
    hit the InferenceService directly) — a caller that wants predictions
    logged calls `log_prediction` itself, e.g. via routers/models.py's
    POST /models/{name}/predictions/log.
    """

    @abstractmethod
    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        input_payload: dict[str, object],
        output_payload: dict[str, object] | None,
    ) -> None: ...

    @abstractmethod
    def list_predictions(self, model_name: str, limit: int = 50) -> list[PredictionLogEntry]: ...


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
        index_version: str | None = None,
    ) -> UpsertResult: ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        collection: str | None = None,
        index_version: str | None = None,
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
    # From LiteLLMGatewayAdapter's response header; None if no cost entry configured.
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
        varies by `group_by` ("team"/"customer"/None) — polymorphic dict,
        same as IVersionRegistryAdapter."""
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

    @abstractmethod
    def list_available_features(self) -> list[str]:
        """`<feature_view>:<feature>` references for every feature this
        store actually has — backs the Scaffolder UI's Feature names
        picker so a user chooses from what's real instead of typing one
        blind."""
        ...

    @abstractmethod
    def list_entity_ids(self) -> list[str]:
        """Every entity id the offline store holds — lets the UI warn when a
        dataset's chosen entity column shares no values with the store, so
        the features would silently come back all-NaN."""
        ...


class NotebookSpec(TypedDict):
    """The resource profile a notebook was spawned with — cached by adapters
    whose backend (JupyterHub's user API) doesn't echo it back."""

    environment: str
    cpu_cores: int
    ram_gb: int
    gpu_type: str | None
    gpu_count: int
    storage_gb: int
    idle_timeout_minutes: int
    created_at: str


class NotebookStatus(TypedDict):
    """One notebook's live state + the resource profile it was spawned with.

    Mirrors the Viettel AI Notebooks service: independent notebooks with
    CPU/GPU/RAM/Storage, on/off lifecycle, idle auto-shutdown, and a
    persistent per-user working environment (storage survives stop/start).
    """

    notebook_id: str
    url: str | None
    active: bool
    environment: str
    cpu_cores: int
    ram_gb: int
    gpu_type: str | None
    gpu_count: int
    storage_gb: int
    idle_timeout_minutes: int
    created_at: str
    last_activity_at: str | None


class NotebookDeletion(TypedDict):
    notebook_id: str
    deleted: bool


class INotebookAdapter(ABC):
    """AI Notebook (JupyterHub) lifecycle. The Portal only provisions and
    manages notebooks — the editing surface stays JupyterHub's own web IDE,
    never re-implemented in Backstage.

    `create_notebook`'s resource args are keyword-only and defaulted so the
    original `(environment, ram_gb, gpu_type)` call shape keeps working.
    """

    @abstractmethod
    def create_notebook(
        self,
        environment: str,
        ram_gb: int,
        gpu_type: str | None = None,
        *,
        cpu_cores: int = 2,
        gpu_count: int = 1,
        storage_gb: int = 20,
        idle_timeout_minutes: int = 60,
    ) -> NotebookStatus: ...

    @abstractmethod
    def get_notebook_status(self, notebook_id: str) -> NotebookStatus: ...

    @abstractmethod
    def list_notebooks(self) -> list[NotebookStatus]:
        """Every notebook this adapter knows about — backs a future Portal
        management view (status, resource usage, session history)."""
        ...

    @abstractmethod
    def start_notebook(self, notebook_id: str) -> NotebookStatus:
        """Power a stopped notebook back on — its storage/working
        environment persists, so this resumes rather than recreates."""
        ...

    @abstractmethod
    def stop_notebook(self, notebook_id: str) -> NotebookStatus:
        """Power off without deleting — frees CPU/GPU while keeping storage."""
        ...

    @abstractmethod
    def delete_notebook(self, notebook_id: str) -> NotebookDeletion: ...


class DatasetInfo(TypedDict):
    name: str
    uri: str
    size_bytes: int
    source: str


class IObjectStorageAdapter(ABC):
    """Discovers datasets already available for training — backs the
    Scaffolder UI's dataset picker so users choose from what's actually there.

    Two implementations merged by `CompositeObjectStorageAdapter`:
    `MinioObjectStorageAdapter` (MinIO/S3, source="s3") and
    `LocalFileObjectStorageAdapter` (the `data/` working-tree checkout,
    source="local"). `DatasetInfo.uri` is always a `file://` path the training
    pod can open (it reads from a hostPath mount, not the bucket).
    """

    @abstractmethod
    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]: ...

    @abstractmethod
    def read_dataset(self, uri: str) -> bytes:
        """Reads one dataset's raw bytes by the `file://` URI `list_datasets`
        handed out — lets the preview/columns/validate endpoints show what a
        dataset actually contains without assuming it's on this process's
        own filesystem (MinIO/S3 objects aren't)."""
        ...


class HuggingFaceModelInfo(TypedDict):
    """Everything the "Serve LLM (Self-hosted)" wizard's Model Source step
    needs to validate a `huggingFaceModelId` and pre-fill the VRAM estimator,
    before a PR ever opens — fail-fast instead of fail-at-PR-merge.

    `num_*`/`hidden_size` are None when the model is gated and no token was
    configured to read config.json — callers show "unavailable, model is
    gated" rather than guessing.
    """

    model_id: str
    exists: bool
    is_gated: bool
    param_count_billion: float | None
    max_context_length: float | None
    num_layers: int | None
    hidden_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    license: str | None


class IHuggingFaceHubAdapter(ABC):
    @abstractmethod
    def get_model_info(self, model_id: str) -> HuggingFaceModelInfo:
        """Never raises for a model that doesn't exist or is gated —
        `HuggingFaceModelInfo.exists`/`is_gated` carry that instead, so a
        routine "model not found" (a typo, most likely) surfaces as a
        normal field the caller checks, not an exception path."""
        ...

    @abstractmethod
    def search_models(self, query: str, limit: int = 20) -> list[str]:
        """Model ids matching `query`, most-downloaded first — backs the
        form's autocomplete so a Dev picks a real id instead of typing one.
        An empty query returns the most popular models."""
        ...


class IEvalResultAdapter(ABC):
    """Persists and queries LLM-as-a-judge evaluation results for LLMOps
    (prompts and RAG indexes). Used by orchestration-api to log gate
    evaluation outcomes and to find the last failure for MTTR calculation."""

    @abstractmethod
    def log_judge_result(
        self,
        kind: str,
        name: str,
        version: str,
        judge_result: object,
        passed: bool,
    ) -> None:
        """Logs a judge evaluation result.

        Args:
            kind: "prompt" | "rag-index"
            name: The prompt or collection name
            version: The version string
            judge_result: The raw judge result (safety, correctness, relevance, reasoning)
            passed: Whether the gate evaluation passed
        """
        ...

    @abstractmethod
    def get_last_failure_at(self, kind: str, name: str) -> datetime | None:
        """Timestamp of the most recent failed evaluation for kind/name, or
        None if none — used for MTTR (time from failure to remediation)."""
        ...


class CostLedgerEntry(TypedDict):
    """One append-only cost event, attributed to an AI artifact's lifecycle.

    Every golden-path step that spends money (train, RAG ingest, eval judge,
    deploy, serve) writes one of these, so cost follows the artifact rather
    than the K8s topology. `stage` is what lets the dashboard separate
    one-time Build cost from recurring Run cost.
    """

    # ISO 8601 instant the cost was incurred.
    timestamp: str
    # "build" | "gate" | "run" — see CostStage in the frontend.
    stage: str
    # "model" | "prompt" | "rag-index" | "eval-set" | "service".
    artifact_kind: str
    artifact_id: str
    version: str
    environment: str
    # Catalog namespace the cost is attributed to (the observer cost API is
    # queried per namespace, so entries must carry it to be matched).
    namespace: str
    # Attribution dimensions, mirroring the golden-path forms.
    team: str
    business_domain: str
    # What was consumed and at what unit price, so the ledger stays auditable.
    quantity: float
    unit: str
    unit_price: float
    cost_usd: float
    # Where the number came from: "litellm" | "mlflow" | "observer" | "gpu-pricing".
    source: str
    # Correlates every entry a single template run produced.
    run_id: str


class ChatDraft(TypedDict):
    """An agent-proposed set of Golden Path form values, accumulated across
    turns. `template` is the golden path name; `form_data` is the flat
    field->value map the frontend seeds the Scaffolder form with."""

    template: str
    form_data: dict[str, object]
    updated_at: str


class ChatSession(TypedDict):
    """One chat conversation's server-side state: the message history (for
    reload coherence) plus the current template draft (if any). Keyed by a
    client-generated `session_id`; `user_ref` scopes reads to the owner."""

    session_id: str
    user_ref: str
    messages: list[dict[str, str]]
    draft: ChatDraft | None
    updated_at: str


class IChatSessionStore(ABC):
    """Server-side store for chat sessions — message history + template
    draft, so a reload restores both. Local-file backed (SQLite) for the
    same reason the cost ledger is: a demo needs it to survive a restart
    without standing up a database; swapping to Postgres later is one new
    class implementing this interface."""

    @abstractmethod
    def get(self, session_id: str) -> ChatSession | None: ...

    @abstractmethod
    def append_messages(
        self, session_id: str, user_ref: str, messages: Sequence[Mapping[str, str]]
    ) -> None:
        """Append turns to the session, creating it if absent."""
        ...

    @abstractmethod
    def merge_draft(
        self, session_id: str, user_ref: str, template: str, patch: Mapping[str, object]
    ) -> ChatDraft:
        """Merge `patch` into the session's draft (not a blind overwrite),
        switching `template` if it changed. Returns the merged draft."""
        ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...


class ICostAdapter(ABC):
    """Append-only cost ledger + aggregation, backing the Cost Insights page.

    Deliberately source-agnostic: the concrete adapter may read LiteLLM's
    spend ledger, MLflow run compute, the OpenChoreo observer's infra cost, or
    a static GPU price table — callers only ever see attributed entries.
    """

    @abstractmethod
    def record_cost(self, entry: CostLedgerEntry) -> None:
        """Append one cost event. Never mutates an existing entry."""
        ...

    @abstractmethod
    def query_costs(
        self,
        start_time: str,
        end_time: str,
        *,
        stage: str | None = None,
        artifact_kind: str | None = None,
        team: str | None = None,
        business_domain: str | None = None,
        environment: str | None = None,
        namespace: str | None = None,
    ) -> list[CostLedgerEntry]:
        """Entries in [start_time, end_time], narrowed by any provided filter."""
        ...
