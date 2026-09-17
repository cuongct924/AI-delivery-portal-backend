"""Single place that decides which concrete class backs each Adapter —
swapping Mock -> real backend means editing this file, not every caller.

`@lru_cache` makes each getter a process-wide singleton. Model
Registry/Workflow/Inference/Notebook each get a `USE_MOCK_<ADAPTER>` flag
(see .env.example), falling back to the blanket `USE_MOCK_ADAPTERS` when
unset.

Workflow and Inference additionally accept a 3rd mode, "openchoreo" (via
`_backend_mode`, only selectable through the specific env var, never the
blanket one) — both real now (OpenChoreoWorkflowAdapter,
OpenChoreoInferenceAdapter). Workflow finished migrating, so its
unset-specific-var default is "openchoreo" and the old Argo Server backend
(`USE_MOCK_WORKFLOW=legacy`) is gone — asking for it now raises instead of
silently running the removed path. Inference hasn't migrated, so it still
defaults to "legacy". Model Registry and Notebook have no OpenChoreo-backed
replacement planned, so they stay on the plain 2-way `_use_mock`.
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
    """`specific_env_var` (e.g. "USE_MOCK_INFERENCE") wins when set; unset
    falls back to the blanket USE_MOCK_ADAPTERS default. Lets a demo mock
    just one adapter (USE_MOCK_INFERENCE=true) while every other adapter
    stays on its real backend, without needing USE_MOCK_ADAPTERS=true to
    mock all 4 at once. An empty string (e.g. an unfilled-in .env template
    value like "USE_MOCK_NOTEBOOK=") counts as unset, not as an explicit
    "false" — otherwise it silently defeats the USE_MOCK_ADAPTERS fallback
    for whichever adapters the .env template happened to list."""
    specific = os.getenv(specific_env_var)
    if specific:
        return specific.lower() == "true"
    return os.getenv("USE_MOCK_ADAPTERS", "false").lower() == "true"


type BackendMode = Literal["mock", "legacy", "openchoreo"]


def _backend_mode(specific_env_var: str, real_default: BackendMode = "legacy") -> BackendMode:
    """Same fallback semantics as `_use_mock`: `specific_env_var` (e.g.
    "USE_MOCK_WORKFLOW") wins when set — accepts "mock"/"legacy"/
    "openchoreo" or the legacy "true"/"false" (back-compat with existing
    .env files). Unset falls back to `USE_MOCK_ADAPTERS`: true -> "mock",
    otherwise the caller's `real_default` — "openchoreo" is never selected
    via the blanket flag, only the specific one or a caller opting in.
    An empty string counts as unset — same reasoning as `_use_mock`'s own
    docstring.

    `real_default` lets an adapter that has finished migrating (Workflow)
    make OpenChoreo the real default, while an unmigrated one (Inference)
    keeps resolving to "legacy". A caller may reject a mode it no longer
    supports — `get_workflow_adapter()` raises on "legacy".
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
    """Cheap, side-effect-free version of the decision `get_kserve_adapter`
    makes internally — unlike that function, never constructs a real
    adapter (KServeAdapter/OpenChoreoInferenceAdapter's `__init__` isn't
    free, both eagerly call `kubernetes.config.load_kube_config()`). For a
    caller that needs to know WHICH backend is active before it needs a
    working adapter instance — e.g. routers/models.py's
    prepare_deploy_manifest picking which manifest template/file path a
    PR-gated deploy renders, without needing cluster access just to render
    text."""
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
    # No mock/real split — SQLite has no external service to fake out
    # (same reasoning as get_registry_adapter's JsonFileVersionRegistryAdapter
    # never getting one either).
    return SqlitePredictionLogAdapter()


@lru_cache
def get_promotion_adapter() -> IPromotionAdapter:
    # No "legacy" branch — the Kargo-based promotion pipeline this would
    # have replaced was removed before it was ever wired up (see
    # OpenChoreoPromotionAdapter's own module docstring), so this is
    # Mock/real only, same 2-way split as get_model_registry_adapter.
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
    # Golden Paths #1/#3 are fully cut over — OpenChoreo is the real (and
    # only) production backend; the Argo Server adapter was removed.
    match _backend_mode("USE_MOCK_WORKFLOW", real_default="openchoreo"):
        case "mock":
            # Only wired to the registry when that's also mocked — a real
            # registry gets its models registered by whatever actually
            # trained them (the ClusterWorkflow's register-step), same as
            # production.
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
    """Real branch is never cached (KServeAdapter.__init__ loads a real
    kubeconfig). Mock branch is cached per tenant so a model's "deployed"
    state persists across calls within a demo run.

    "legacy" always targets `ai-delivery-portal-dev-<tenant>` — staging/prod
    promotion is DeploymentPipeline-only (infra/openchoreo/deployment-pipeline.yaml).
    "openchoreo" ignores `tenant` entirely: it's scoped to the one real
    serving Component this repo has (`serving`, under the
    `telco-fraud-detection` Project), not a
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
