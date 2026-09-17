"""Single place deciding which concrete class backs each Adapter — swapping
Mock -> real backend means editing this file, not every caller.

`@lru_cache` makes each getter a process-wide singleton. Each adapter gets a
`USE_MOCK_<NAME>` flag (see .env.example), falling back to the blanket
`USE_MOCK_ADAPTERS` when unset.

Workflow and Inference add a 3rd mode, "openchoreo", via `_backend_mode`
(only selectable through the specific env var). Workflow finished migrating —
its unset default is "openchoreo" and the old Argo backend (USE_MOCK_WORKFLOW=legacy)
raises. Inference hasn't migrated, so it still defaults to "legacy". Model
Registry and Notebook stay on the plain 2-way `_use_mock`.
"""

import os
from functools import lru_cache
from typing import Literal, cast

from adapters.composite_object_storage_adapter import CompositeObjectStorageAdapter
from adapters.feature_store_adapter import FeastAdapter
from adapters.huggingface_hub_adapter import HuggingFaceHubAdapter
from adapters.interfaces import (
    IEvalResultAdapter,
    IHuggingFaceHubAdapter,
    ILLMGatewayAdapter,
    IObjectStorageAdapter,
    IPredictionLogAdapter,
    IPromotionAdapter,
    IVersionRegistryAdapter,
)
from adapters.kserve_adapter import KServeAdapter
from adapters.llm_gateway_adapter import LiteLLMGatewayAdapter
from adapters.local_object_storage_adapter import LocalFileObjectStorageAdapter
from adapters.mlflow_adapter import MlflowAdapter
from adapters.mlflow_eval_result_adapter import MlflowEvalResultAdapter
from adapters.mock_eval_result_adapter import MockEvalResultAdapter
from adapters.mock_huggingface_hub_adapter import MockHuggingFaceHubAdapter
from adapters.mock_inference_adapter import MockInferenceAdapter
from adapters.mock_model_registry_adapter import MockModelRegistryAdapter
from adapters.mock_notebook_adapter import MockNotebookAdapter
from adapters.mock_promotion_adapter import MockPromotionAdapter
from adapters.mock_workflow_adapter import MockWorkflowAdapter
from adapters.notebook_adapter import JupyterHubAdapter
from adapters.object_storage_adapter import MinioObjectStorageAdapter
from adapters.openchoreo_inference_adapter import OpenChoreoInferenceAdapter
from adapters.openchoreo_promotion_adapter import OpenChoreoPromotionAdapter
from adapters.openchoreo_workflow_adapter import OpenChoreoWorkflowAdapter
from adapters.prediction_log_adapter import SqlitePredictionLogAdapter
from adapters.prompt_registry_adapter import MlflowPromptRegistryAdapter
from adapters.vector_db_adapter import QdrantAdapter
from adapters.version_registry_adapter import JsonFileVersionRegistryAdapter


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
    """Side-effect-free version of `get_kserve_adapter`'s decision — never
    constructs an adapter (both real `__init__`s eagerly load kubeconfig). For
    callers that need to know the backend before they need an instance, e.g.
    routers/models.py picking the PR render template."""
    return _backend_mode("USE_MOCK_INFERENCE")


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
def get_promotion_adapter() -> IPromotionAdapter:
    # No "legacy" branch — the Kargo-based pipeline this would have replaced
    # was removed before ever being wired up.
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
    # Golden Paths #1/#3 are fully cut over — OpenChoreo is the only
    # production backend; the Argo Server adapter was removed.
    match _backend_mode("USE_MOCK_WORKFLOW", real_default="openchoreo"):
        case "mock":
            # Only wired to the registry when that's also mocked — a real
            # registry gets its models registered by whatever actually
            # trained them (the ClusterWorkflow's register-step).
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


_mock_kserve_adapters: dict[str, MockInferenceAdapter] = {}


def get_kserve_adapter(
    tenant: str,
) -> KServeAdapter | MockInferenceAdapter | OpenChoreoInferenceAdapter:
    """Real branch is never cached (KServeAdapter.__init__ loads kubeconfig);
    mock branch is cached per tenant so a model's "deployed" state persists
    within a demo run.

    "legacy" targets `ai-delivery-portal-dev-<tenant>` — staging/prod is
    DeploymentPipeline-only. "openchoreo" ignores `tenant`: it's scoped to the
    one real serving Component (`serving`, telco-fraud-detection), not a
    per-tenant namespace — see OpenChoreoInferenceAdapter's docstring.
    """
    namespace = f"ai-delivery-portal-dev-{tenant}"
    match _backend_mode("USE_MOCK_INFERENCE"):
        case "mock":
            if tenant not in _mock_kserve_adapters:
                _mock_kserve_adapters[tenant] = MockInferenceAdapter(namespace=namespace)
            return _mock_kserve_adapters[tenant]
        case "legacy":
            return KServeAdapter(namespace=namespace)
        case "openchoreo":
            return OpenChoreoInferenceAdapter()
