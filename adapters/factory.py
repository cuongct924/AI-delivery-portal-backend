"""Single place deciding which concrete class backs each Adapter — swapping
Mock -> real backend means editing this file, not every caller.

`@lru_cache` makes each getter a process-wide singleton. Each adapter gets a
`USE_MOCK_<NAME>` flag (see .env.example), falling back to the blanket
`USE_MOCK_ADAPTERS` when unset.

Workflow and (standard-serving) Inference add a 3rd mode, "openchoreo", via
`_backend_mode` (only selectable through the specific env var). Both have
finished migrating — their unset default is "openchoreo" and the old
real-backend value (USE_MOCK_WORKFLOW=legacy / USE_MOCK_INFERENCE=legacy)
raises. GPU inference (LLM self-hosted serving,
get_gpu_inference_adapter()) is a separate getter with its own
USE_MOCK_GPU_INFERENCE flag and no "openchoreo" mode — no ClusterComponentType
for GPU/vLLM serving exists, so it stays plain mock/real (KServe) via
`_use_mock`, same as Model Registry and Notebook.
"""

import os
from functools import lru_cache
from typing import Literal, cast

from adapters.ai_platform.cost_adapter import JsonFileCostLedgerAdapter
from adapters.ai_platform.feature_store_adapter import FeastAdapter
from adapters.ai_platform.huggingface_hub_adapter import HuggingFaceHubAdapter
from adapters.ai_platform.interfaces import (
    ICostAdapter,
    IEvalResultAdapter,
    IHuggingFaceHubAdapter,
    ILLMGatewayAdapter,
    IObjectStorageAdapter,
    IPredictionLogAdapter,
    IVersionRegistryAdapter,
)
from adapters.ai_platform.llm_gateway_adapter import LiteLLMGatewayAdapter
from adapters.ai_platform.mlflow_adapter import MlflowAdapter
from adapters.ai_platform.mlflow_eval_result_adapter import MlflowEvalResultAdapter
from adapters.ai_platform.mock_misc_adapters import (
    MockEvalResultAdapter,
    MockHuggingFaceHubAdapter,
    MockNotebookAdapter,
)
from adapters.ai_platform.mock_model_registry_adapter import MockModelRegistryAdapter
from adapters.ai_platform.notebook_adapter import JupyterHubAdapter
from adapters.ai_platform.object_storage import (
    CompositeObjectStorageAdapter,
    LocalFileObjectStorageAdapter,
    MinioObjectStorageAdapter,
)
from adapters.ai_platform.prediction_log_adapter import SqlitePredictionLogAdapter
from adapters.ai_platform.prompt_registry_adapter import MlflowPromptRegistryAdapter
from adapters.ai_platform.vector_db_adapter import QdrantAdapter
from adapters.ai_platform.version_registry_adapter import JsonFileVersionRegistryAdapter
from adapters.delivery.deployment_event_store import SqliteDeploymentEventStore
from adapters.delivery.gpu_inference_adapter import GpuKServeInferenceAdapter
from adapters.delivery.interfaces import (
    IDeliveryObserverAdapter,
    IGpuInferenceAdapter,
    IInferenceAdapter,
    IPromotionAdapter,
)
from adapters.delivery.mock_delivery_observer_adapter import MockDeliveryObserverAdapter
from adapters.delivery.mock_inference_adapter import MockInferenceAdapter
from adapters.delivery.mock_promotion_adapter import MockPromotionAdapter
from adapters.delivery.mock_workflow_adapter import MockWorkflowAdapter
from adapters.delivery.openchoreo_inference_adapter import OpenChoreoInferenceAdapter
from adapters.delivery.openchoreo_promotion_adapter import OpenChoreoPromotionAdapter
from adapters.delivery.openchoreo_workflow_adapter import OpenChoreoWorkflowAdapter
from adapters.delivery.prometheus_delivery_observer_adapter import PrometheusDeliveryObserverAdapter


def _use_mock(specific_env_var: str) -> bool:
    """`specific_env_var` (e.g. "USE_MOCK_INFERENCE") wins when set; unset falls
    back to the blanket USE_MOCK_ADAPTERS. An empty string (e.g. a blank .env
    template value) counts as unset, so it can't silently defeat the fallback."""
    specific = os.getenv(specific_env_var)
    if specific:
        return specific.lower() == "true"
    return os.getenv("USE_MOCK_ADAPTERS", "false").lower() == "true"


type BackendMode = Literal["mock", "legacy", "openchoreo"]


def _backend_mode(specific_env_var: str, real_default: BackendMode = "legacy") -> BackendMode:
    """Same fallback semantics as `_use_mock`, plus "openchoreo": accepts
    "mock"/"legacy"/"openchoreo" or legacy "true"/"false" (back-compat with
    existing .env files). Unset falls back to `USE_MOCK_ADAPTERS` (true ->
    "mock") or `real_default` — "openchoreo" is never picked via the blanket
    flag. `real_default` lets a migrated adapter (Workflow) default to
    OpenChoreo while an unmigrated one (Inference) keeps resolving to
    "legacy"; a caller may reject a mode it no longer supports.
    """
    specific = os.getenv(specific_env_var)
    if specific:
        normalized = specific.lower()
        if normalized in ("mock", "legacy", "openchoreo"):
            return cast(BackendMode, normalized)
        return "mock" if normalized == "true" else real_default
    if os.getenv("USE_MOCK_ADAPTERS", "false").lower() == "true":
        return "mock"
    return real_default


def get_inference_backend_mode() -> BackendMode:
    """Side-effect-free version of `get_inference_adapter`'s decision — never
    constructs an adapter (the real `__init__` eagerly loads kubeconfig). For
    callers that need to know the backend before they need an instance, e.g.
    routers/models.py picking the PR render template."""
    return _backend_mode("USE_MOCK_INFERENCE", real_default="openchoreo")


@lru_cache
def get_llm_gateway_adapter() -> ILLMGatewayAdapter:
    return LiteLLMGatewayAdapter()


@lru_cache
def get_vector_store_adapter() -> QdrantAdapter:
    return QdrantAdapter()


@lru_cache
def get_huggingface_hub_adapter() -> IHuggingFaceHubAdapter:
    if _use_mock("USE_MOCK_HUGGINGFACE"):
        return MockHuggingFaceHubAdapter()
    return HuggingFaceHubAdapter()


@lru_cache
def get_registry_adapter() -> IVersionRegistryAdapter:
    """Backs kind="rag-index" (routers/rag.py). kind="prompt" moved to
    get_prompt_registry_adapter() below — see that adapter's docstring for
    why the two kinds don't share one backend."""
    return JsonFileVersionRegistryAdapter()


@lru_cache
def get_prompt_registry_adapter() -> IVersionRegistryAdapter:
    return MlflowPromptRegistryAdapter()


@lru_cache
def get_prediction_log_adapter() -> IPredictionLogAdapter:
    # No mock/real split — SQLite has no external service to fake out.
    return SqlitePredictionLogAdapter()


@lru_cache
def get_cost_adapter() -> ICostAdapter:
    # No mock/real split — the ledger is a local append-only file, and the
    # sources it aggregates (LiteLLM/MLflow/observer) are read by the callers
    # that record entries, not by this adapter.
    return JsonFileCostLedgerAdapter()


@lru_cache
def get_promotion_adapter() -> IPromotionAdapter:
    # No "legacy" branch — the Kargo pipeline was removed before ever wiring up.
    if _use_mock("USE_MOCK_PROMOTION"):
        return MockPromotionAdapter()
    return OpenChoreoPromotionAdapter()


@lru_cache
def get_model_registry_adapter() -> MlflowAdapter | MockModelRegistryAdapter:
    if _use_mock("USE_MOCK_MODEL_REGISTRY"):
        return MockModelRegistryAdapter()
    return MlflowAdapter()


@lru_cache
def get_workflow_adapter() -> MockWorkflowAdapter | OpenChoreoWorkflowAdapter:
    # Golden Paths #1/#3 fully cut over — OpenChoreo is the only backend now.
    match _backend_mode("USE_MOCK_WORKFLOW", real_default="openchoreo"):
        case "mock":
            # Only wired when the registry is also mocked — real training registers itself.
            model_registry = get_model_registry_adapter()
            return MockWorkflowAdapter(
                model_registry=model_registry
                if isinstance(model_registry, MockModelRegistryAdapter)
                else None
            )
        case "openchoreo":
            return OpenChoreoWorkflowAdapter()
        case "legacy":
            raise ValueError(
                "USE_MOCK_WORKFLOW=legacy is no longer supported: adapters/argo_adapter.py "
                "was removed when Golden Paths #1/#3 moved to OpenChoreo. Set it to "
                "'openchoreo' (real) or 'mock'."
            )


@lru_cache
def get_notebook_adapter() -> JupyterHubAdapter | MockNotebookAdapter:
    if _use_mock("USE_MOCK_NOTEBOOK"):
        return MockNotebookAdapter()
    return JupyterHubAdapter()


@lru_cache
def get_feature_store_adapter() -> FeastAdapter:
    return FeastAdapter()


@lru_cache
def get_object_storage_adapter() -> IObjectStorageAdapter:
    """Local checked-out files first (fast, no network), then S3/MinIO —
    order only affects the picker's display order, both sources always
    get listed."""
    return CompositeObjectStorageAdapter(
        [LocalFileObjectStorageAdapter(), MinioObjectStorageAdapter()]
    )


@lru_cache
def get_eval_result_adapter() -> IEvalResultAdapter:
    if _use_mock("USE_MOCK_EVAL_RESULT"):
        return MockEvalResultAdapter()
    return MlflowEvalResultAdapter()


@lru_cache
def get_deployment_event_store() -> SqliteDeploymentEventStore:
    # No mock/real split — SQLite has no external service to fake out.
    return SqliteDeploymentEventStore()


@lru_cache
def get_delivery_observer_adapter() -> IDeliveryObserverAdapter:
    if _use_mock("USE_MOCK_DELIVERY_OBSERVER"):
        return MockDeliveryObserverAdapter()
    return PrometheusDeliveryObserverAdapter(
        model_registry_adapter=get_model_registry_adapter(),
        deployment_event_store=get_deployment_event_store(),
    )


_mock_inference_adapters: dict[str, MockInferenceAdapter] = {}


def get_inference_adapter(tenant: str) -> IInferenceAdapter:
    """Standard model-serving golden path (routers/models.py). mock |
    openchoreo only — the real "legacy" (generic KServe-direct) backend was
    removed once this golden path fully migrated to OpenChoreo; selecting it
    now raises, mirroring get_workflow_adapter()'s "legacy" precedent.

    Real branch is never cached (OpenChoreoInferenceAdapter.__init__ loads
    kubeconfig); mock branch is cached per tenant so a model's "deployed"
    state persists within a demo run. "openchoreo" ignores `tenant`: it's
    scoped to the one real serving Component (`serving`,
    telco-fraud-detection), not a per-tenant namespace — see
    OpenChoreoInferenceAdapter's docstring.
    """
    match _backend_mode("USE_MOCK_INFERENCE", real_default="openchoreo"):
        case "mock":
            namespace = f"ai-delivery-portal-dev-{tenant}"
            if tenant not in _mock_inference_adapters:
                _mock_inference_adapters[tenant] = MockInferenceAdapter(namespace=namespace)
            return _mock_inference_adapters[tenant]
        case "openchoreo":
            return OpenChoreoInferenceAdapter()
        case "legacy":
            raise ValueError(
                "USE_MOCK_INFERENCE=legacy is no longer supported: the standard "
                "model-serving golden path moved fully to OpenChoreo, and the generic "
                "KServe-direct deploy_model() was removed from "
                "adapters/delivery/gpu_inference_adapter.py (now GPU-serving-only). "
                "Set it to 'openchoreo' (real) or 'mock'."
            )


_mock_gpu_inference_adapters: dict[str, MockInferenceAdapter] = {}


def get_gpu_inference_adapter(tenant: str) -> IGpuInferenceAdapter:
    """LLM self-hosted-serving golden path (routers/llm_serving.py). mock |
    real (KServe) only — no "openchoreo" mode exists, since no
    ClusterComponentType for GPU/vLLM serving exists yet. Separate flag
    (USE_MOCK_GPU_INFERENCE) from get_inference_adapter's USE_MOCK_INFERENCE
    — one flag can no longer sanely govern two structurally different
    backends now that they've diverged.

    Real branch is never cached (GpuKServeInferenceAdapter.__init__ loads
    kubeconfig); mock branch is cached per tenant so a model's "deployed"
    state persists within a demo run. Targets
    `ai-delivery-portal-dev-<tenant>` — staging/prod is
    DeploymentPipeline-only (see prepare_llm_deploy_manifest's docstring).
    """
    namespace = f"ai-delivery-portal-dev-{tenant}"
    if _use_mock("USE_MOCK_GPU_INFERENCE"):
        if tenant not in _mock_gpu_inference_adapters:
            _mock_gpu_inference_adapters[tenant] = MockInferenceAdapter(namespace=namespace)
        return _mock_gpu_inference_adapters[tenant]
    return GpuKServeInferenceAdapter(namespace=namespace)
