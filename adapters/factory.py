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
"""

from functools import lru_cache

from adapters.argo_adapter import ArgoAdapter
from adapters.feature_store_adapter import FeastAdapter
from adapters.interfaces import ILLMGatewayAdapter, IObjectStorageAdapter, IVersionRegistryAdapter
from adapters.kserve_adapter import KServeAdapter
from adapters.llm_gateway_adapter import LiteLLMGatewayAdapter
from adapters.mlflow_adapter import MlflowAdapter
from adapters.object_storage_adapter import MinioObjectStorageAdapter
from adapters.prompt_registry_adapter import MlflowPromptRegistryAdapter
from adapters.vector_db_adapter import QdrantAdapter
from adapters.version_registry_adapter import JsonFileVersionRegistryAdapter


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
def get_model_registry_adapter() -> MlflowAdapter:
    return MlflowAdapter()


@lru_cache
def get_workflow_adapter() -> ArgoAdapter:
    return ArgoAdapter()


@lru_cache
def get_feature_store_adapter() -> FeastAdapter:
    return FeastAdapter()


@lru_cache
def get_object_storage_adapter() -> IObjectStorageAdapter:
    return MinioObjectStorageAdapter()


def get_kserve_adapter(tenant: str) -> KServeAdapter:
    """Not cached — KServeAdapter.__init__ loads a real kubeconfig, so
    callers only construct it when a request actually needs KServe
    (see routers/models.py, routers/llm_serving.py), never eagerly at
    import time.

    Always targets `ai-delivery-portal-dev-<tenant>` — orchestration-api
    only ever writes/deploys into dev (staging/prod are Kargo-only, see
    infra/kargo/README.md), so there is no code path here that can target
    any other namespace, which is what keeps `release_strategy=instant`
    (adapters/deploy_strategies.py's InstantStrategy) from being able to
    bypass Kargo's approval gate.
    """
    return KServeAdapter(namespace=f"ai-delivery-portal-dev-{tenant}")
