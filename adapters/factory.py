"""Single place that decides which concrete class backs each Adapter.
Callers import the getter (not the concrete class) — swapping Mock -> a
real backend means editing exactly this file, not every router that uses
it.

`@lru_cache` makes each getter a process-wide singleton, same lifecycle as
the module-level instances routers used to construct directly.

Getters are typed against the interface (`I*Adapter`) except where a real
caller needs a "convenience method" the interface deliberately excludes
(ArgoAdapter.create_cron_workflow/list_workflows,
MlflowAdapter.get_latest_version, KServeAdapter.deploy_llm_model,
QdrantAdapter.ensure_collection — see each adapter's own docstring) —
those stay typed to the concrete class rather than adding backend-specific
methods to a shared interface.

Each of Model Registry/Workflow/Inference/Notebook has its own
`USE_MOCK_<ADAPTER>` flag (see .env.example) swapping it for its in-memory
Mock counterpart — a stand-in for a backend this repo doesn't self-host yet
(a real cluster for KServe/Argo) or a product a separate team owns (see
docs/playbook-ai-delivery-portal.md's AI Platform box). `USE_MOCK_ADAPTERS`
sets the default for whichever of those 4 flags isn't itself set — e.g.
scripts/local-demo/fake_argo.py stands in for the real Argo Server on the
wire (ArgoAdapter still runs for real, just talks to a fake backend, so
Golden Path #1 logs a real trained model to MLflow), while nothing stands
in for a real Kubernetes API server, so USE_MOCK_INFERENCE=true is the only
way to demo Golden Path #2 without a `kind` cluster. Everything else (LLM
Gateway, Vector Store, prompt/RAG registry, Object Storage, Feature Store)
already has a real local backend via docker-compose.yml, so it's never
mocked. Callers are unaffected either way — each Mock class mirrors its
real counterpart's full method set (interface + convenience methods).
"""

import os
from functools import lru_cache

from adapters.argo_adapter import ArgoAdapter
from adapters.feature_store_adapter import FeastAdapter
from adapters.interfaces import ILLMGatewayAdapter, IObjectStorageAdapter, IVersionRegistryAdapter
from adapters.kserve_adapter import KServeAdapter
from adapters.llm_gateway_adapter import LiteLLMGatewayAdapter
from adapters.mlflow_adapter import MlflowAdapter
from adapters.mock_inference_adapter import MockInferenceAdapter
from adapters.mock_model_registry_adapter import MockModelRegistryAdapter
from adapters.mock_notebook_adapter import MockNotebookAdapter
from adapters.mock_workflow_adapter import MockWorkflowAdapter
from adapters.notebook_adapter import JupyterHubAdapter
from adapters.object_storage_adapter import MinioObjectStorageAdapter
from adapters.prompt_registry_adapter import MlflowPromptRegistryAdapter
from adapters.vector_db_adapter import QdrantAdapter
from adapters.version_registry_adapter import JsonFileVersionRegistryAdapter


def _use_mock(specific_env_var: str) -> bool:
    """`specific_env_var` (e.g. "USE_MOCK_INFERENCE") wins when set; unset
    falls back to the blanket USE_MOCK_ADAPTERS default. Lets a demo mock
    just one adapter (USE_MOCK_INFERENCE=true) while every other adapter
    stays on its real backend, without needing USE_MOCK_ADAPTERS=true to
    mock all 4 at once."""
    specific = os.getenv(specific_env_var)
    if specific is not None:
        return specific.lower() == "true"
    return os.getenv("USE_MOCK_ADAPTERS", "false").lower() == "true"


@lru_cache
def get_llm_gateway_adapter() -> ILLMGatewayAdapter:
    return LiteLLMGatewayAdapter()


@lru_cache
def get_vector_store_adapter() -> QdrantAdapter:
    return QdrantAdapter()


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
def get_model_registry_adapter() -> MlflowAdapter | MockModelRegistryAdapter:
    if _use_mock("USE_MOCK_MODEL_REGISTRY"):
        return MockModelRegistryAdapter()
    return MlflowAdapter()


@lru_cache
def get_workflow_adapter() -> ArgoAdapter | MockWorkflowAdapter:
    if _use_mock("USE_MOCK_WORKFLOW"):
        # Wired to the model registry only when THAT'S also mocked — lets
        # MockWorkflowAdapter register a model after "training" (see its
        # _maybe_register_model()'s docstring) so a Golden Path #1 run
        # entirely under USE_MOCK_ADAPTERS=true doesn't 500 on the very
        # next call. When the registry is real (MlflowAdapter), whatever
        # actually did the training (e.g. scripts/local-demo/fake_argo.py)
        # owns registration instead, same as production.
        model_registry = get_model_registry_adapter()
        return MockWorkflowAdapter(
            model_registry=model_registry
            if isinstance(model_registry, MockModelRegistryAdapter)
            else None
        )
    return ArgoAdapter()


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
    return MinioObjectStorageAdapter()


_mock_kserve_adapters: dict[str, MockInferenceAdapter] = {}


def get_kserve_adapter(tenant: str) -> KServeAdapter | MockInferenceAdapter:
    """Real branch is never cached — KServeAdapter.__init__ loads a real
    kubeconfig, so callers only construct it when a request actually needs
    KServe (see routers/models.py, routers/llm_serving.py), never eagerly at
    import time. The Mock branch has no such cost, and — unlike the real
    cluster, which is the same backend no matter which Python object talks
    to it — its "deployed" state lives on the instance, so it's cached per
    tenant in `_mock_kserve_adapters` instead: a fresh instance per call
    would make Golden Path #2's traffic-split/canary steps see every model
    as never having a prior deploy, even one this same demo run just made.

    Always targets `ai-delivery-portal-dev-<tenant>` — orchestration-api
    only ever writes/deploys into dev (staging/prod promotion is
    DeploymentPipeline-only, see infra/openchoreo/deployment-pipeline.yaml),
    so there is no code path here that can target
    any other namespace, which is what keeps `release_strategy=instant`
    (adapters/deploy_strategies.py's InstantStrategy) from being able to
    bypass Kargo's approval gate.
    """
    namespace = f"ai-delivery-portal-dev-{tenant}"
    if _use_mock("USE_MOCK_INFERENCE"):
        if tenant not in _mock_kserve_adapters:
            _mock_kserve_adapters[tenant] = MockInferenceAdapter(namespace=namespace)
        return _mock_kserve_adapters[tenant]
    return KServeAdapter(namespace=namespace)
