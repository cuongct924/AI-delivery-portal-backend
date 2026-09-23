"""Serving LLM API — deploys a self-hosted LLM (by HuggingFace Hub id, not
an MLflow-registered artifact) as a KServe InferenceService via vLLM.
Separate from models.py: no MLflow Model Registry URI, no Evaluate Gate
(an off-the-shelf LLM's weights don't change).

Also backs the "Serve LLM (Self-hosted)" wizard's pre-flight checks (fail
fast at form-time, not at PR-merge time) — GET /llm-deploy/validate-model,
GET /llm-deploy/gpu-recommendation, GET /llm-deploy/rollout-eligibility —
ahead of the actual POST /llm-deploy/prepare below.
"""

from pathlib import Path
from typing import Any, Final

from auth.thunder import get_current_user, user_has_role
from costs.events import record_cost_event
from costs.pricing import gpu_hour_price
from fastapi import APIRouter, Depends, HTTPException
from jinja2 import Environment, FileSystemLoader
from llm_serving.gpu_sizing import estimate_vram_gb, recommend_gpu
from llm_serving.registry import (
    VLLM_QUANTIZATION_ARGS,
    get_llm_serving_runtime,
    validate_gpu_quantization,
    validate_runtime_optimizations,
)
from pydantic import BaseModel

from adapters.delivery.deploy_strategies import BlueGreenStrategy, DirectStrategy, PRGatedStrategy
from adapters.delivery.interfaces import IDeployTrafficStrategy
from adapters.factory import get_gpu_inference_adapter, get_huggingface_hub_adapter

router = APIRouter(tags=["llm-serving"])

huggingface_hub_adapter = get_huggingface_hub_adapter()

_TEMPLATES_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "templates"
_JINJA_ENV: Final[Environment] = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))

_ENVIRONMENTS: Final[frozenset[str]] = frozenset({"dev", "staging", "prod"})

# release_strategy="instant" needs llm-ops-admin — a real server-side gate,
# since hiding the field client-side alone would be security theater.
_LLM_OPS_ADMIN_ROLE: Final[str] = "llm-ops-admin"


class PrepareLlmDeployRequest(BaseModel):
    model_name: str
    huggingface_model_id: str
    runtime: str = "vllm"
    gpu_type: str
    gpu_count: int = 1
    quantization: str = "none"
    max_context_length: int = 4096
    # "direct" | "blue-green" (same shape as models.py).
    traffic_strategy: str = "direct"
    # Share of traffic to the new version; only meaningful for a non-direct
    # strategy. None → 100 (full cutover).
    traffic_percent: int | None = None
    # "pr-gated" | "instant"
    release_strategy: str = "pr-gated"
    # "dev" | "staging" | "prod" — only "dev" may pair with "instant",
    # see prepare_llm_deploy_manifest's docstring for why.
    environment: str = "dev"
    # A K8s Secret name (key "token"), resolved server-side so a plaintext
    # HuggingFace token never passes through this API or a PR. Required when
    # /llm-deploy/validate-model reported is_gated=true.
    hf_token_secret_ref: str | None = None

    # Only a subset is supported per runtime — validate_runtime_optimizations()
    # rejects the rest with a clear error.
    batchingStrategy: str | None = None  # static | dynamic | continuous
    enablePagedAttention: bool | None = None
    enablePrefixCaching: bool | None = None
    speculativeDecoding: str | None = None  # none | ngram | draft-model
    draftModelId: str | None = None
    pipelineParallelSize: int | None = None


class PrepareLlmDeployResponse(BaseModel):
    file_name: str
    content: str
    deployed: bool = False


class HuggingFaceModelValidationResponse(BaseModel):
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


class GpuFitEstimateResponse(BaseModel):
    gpu_type: str
    gpu_count: int
    vram_available_gb: float
    fits: bool


class GpuRecommendationResponse(BaseModel):
    vram_needed_gb: float
    # None only when no catalog entry (llm_serving.gpu_sizing.GPU_VRAM_GB)
    # fits at up to 8-way tensor parallelism — the wizard shows a red
    # "no GPU configuration fits" warning in that case, per the redesign
    # plan's VRAM estimator widget.
    recommended: GpuFitEstimateResponse | None
    estimates: list[GpuFitEstimateResponse]


class RolloutEligibilityResponse(BaseModel):
    model_name: str
    has_prior_deploy: bool


@router.get("/llm-deploy/validate-model", response_model=HuggingFaceModelValidationResponse)
def validate_huggingface_model(
    huggingface_model_id: str, user: dict = Depends(get_current_user)
) -> HuggingFaceModelValidationResponse:
    """Pre-flight check for the wizard's Model Source step: does this
    model id actually exist, is it gated, and what's its architecture
    (feeds GET /llm-deploy/gpu-recommendation below) — surfaced here so a
    typo'd or gated-without-a-token model id fails at form-time instead of
    inside a PR nobody notices is broken until merge."""
    info = huggingface_hub_adapter.get_model_info(huggingface_model_id)
    return HuggingFaceModelValidationResponse(**info)


class HuggingFaceSearchResponse(BaseModel):
    model_ids: list[str]


@router.get("/llm-deploy/search-models", response_model=HuggingFaceSearchResponse)
def search_huggingface_models(
    q: str = "",
    limit: int = 20,
    user: dict = Depends(get_current_user),
) -> HuggingFaceSearchResponse:
    """Autocomplete for the wizard's HuggingFace model id field — a Dev picks
    a real id instead of typing one (typo-prone)."""
    return HuggingFaceSearchResponse(model_ids=huggingface_hub_adapter.search_models(q, limit))


class SecretNamesResponse(BaseModel):
    names: list[str]


@router.get("/secrets", response_model=SecretNamesResponse)
def list_secrets(
    namespace: str = "default", user: dict = Depends(get_current_user)
) -> SecretNamesResponse:
    """K8s Secret names in the namespace — backs the HF token secret dropdown
    so a Dev picks a real Secret instead of typing a name that leaves the pod
    stuck pulling a gated model. Empty list when no cluster is reachable."""
    try:
        from kubernetes import client as k8s_client

        from adapters.delivery._kube_client import load_kube_config_once

        load_kube_config_once()
        core = k8s_client.CoreV1Api()
        secrets: Any = core.list_namespaced_secret(namespace)
        items = getattr(secrets, "items", None) or []

        def _secret_name(secret: Any) -> str | None:
            metadata = getattr(secret, "metadata", None)
            return getattr(metadata, "name", None)

        return SecretNamesResponse(names=sorted(name for s in items if (name := _secret_name(s))))
    except Exception:  # noqa: BLE001 — no cluster, fall back to the text field
        return SecretNamesResponse(names=[])


@router.get("/llm-deploy/gpu-recommendation", response_model=GpuRecommendationResponse)
def get_gpu_recommendation(
    param_count_billion: float,
    quantization: str = "none",
    max_context_length: int = 4096,
    concurrency: int = 1,
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 32,
    user: dict = Depends(get_current_user),
) -> GpuRecommendationResponse:
    """VRAM-sized GPU/quantity suggestion for the wizard's Compute &
    Runtime step. Architecture defaults (num_layers/hidden_size/
    num_attention_heads/num_key_value_heads) describe a generic 7B-class
    model — the frontend should pass the real values
    GET /llm-deploy/validate-model returned for the chosen model instead
    of relying on these when they're available."""
    vram_needed_gb = estimate_vram_gb(
        param_count_billion=param_count_billion,
        quantization=quantization,
        max_context_length=max_context_length,
        concurrency=concurrency,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    estimates = [
        GpuFitEstimateResponse(
            gpu_type=estimate.gpu_type,
            gpu_count=estimate.gpu_count,
            vram_available_gb=estimate.vram_available_gb,
            fits=estimate.fits,
        )
        for estimate in recommend_gpu(vram_needed_gb)
    ]
    recommended = next((estimate for estimate in estimates if estimate.fits), None)
    return GpuRecommendationResponse(
        vram_needed_gb=vram_needed_gb, recommended=recommended, estimates=estimates
    )


@router.get("/llm-deploy/rollout-eligibility", response_model=RolloutEligibilityResponse)
def get_rollout_eligibility(
    model_name: str, user: dict = Depends(get_current_user)
) -> RolloutEligibilityResponse:
    """Whether blue-green may be offered for `model_name` — the Rollout &
    Release step queries this before enabling anything but 'direct',
    mirroring the exact same prior-deploy check prepare_llm_deploy_manifest
    enforces server-side below (fail fast at form-time, not at submit)."""
    gpu_inference_adapter = get_gpu_inference_adapter("llmops-team")
    has_prior_deploy = gpu_inference_adapter.get_deploy_status(model_name)["deployed"]
    return RolloutEligibilityResponse(model_name=model_name, has_prior_deploy=has_prior_deploy)


@router.post("/llm-deploy/prepare", response_model=PrepareLlmDeployResponse)
def prepare_llm_deploy_manifest(
    request: PrepareLlmDeployRequest, user: dict = Depends(get_current_user)
) -> PrepareLlmDeployResponse:
    """Renders (and, for release_strategy="instant", applies) the
    InferenceService manifest.

    environment="dev" is the only value release_strategy="instant" may
    pair with: get_gpu_inference_adapter namespaces to
    "ai-delivery-portal-dev-<tenant>" unconditionally (adapters/factory.py) —
    no backend this repo has today can honor an instant deploy to
    staging/prod, so that combination is rejected here rather than silently
    deploying to the wrong place (or to nowhere). staging/prod therefore
    always render as a PR-gated manifest for a human to apply through the
    normal promotion path, regardless of what release_strategy was
    requested for them.
    """
    if request.environment not in _ENVIRONMENTS:
        raise ValueError(
            f"unknown environment {request.environment!r} — must be one of {sorted(_ENVIRONMENTS)}"
        )
    if request.release_strategy == "instant":
        if request.environment != "dev":
            raise ValueError(
                "release_strategy='instant' is only possible for environment='dev' — "
                "no backend this repo has today can deploy straight to staging/prod"
            )
        if not user_has_role(user, _LLM_OPS_ADMIN_ROLE):
            raise HTTPException(
                403, f"release_strategy='instant' requires the {_LLM_OPS_ADMIN_ROLE!r} role"
            )

    validate_gpu_quantization(request.gpu_type, request.quantization)
    runtime_spec = get_llm_serving_runtime(request.runtime)
    vllm_quantization = VLLM_QUANTIZATION_ARGS.get(request.quantization)

    optimizations = {
        "batchingStrategy": request.batchingStrategy,
        "enablePagedAttention": request.enablePagedAttention,
        "enablePrefixCaching": request.enablePrefixCaching,
        "speculativeDecoding": request.speculativeDecoding,
        "draftModelId": request.draftModelId,
        "pipelineParallelSize": request.pipelineParallelSize,
    }
    validate_runtime_optimizations(request.runtime, optimizations)

    # Lazy: GpuKServeInferenceAdapter.__init__ eagerly calls load_kube_config().
    needs_gpu_inference_adapter = (
        request.traffic_strategy != "direct" or request.release_strategy == "instant"
    )
    gpu_inference_adapter = (
        get_gpu_inference_adapter("llmops-team") if needs_gpu_inference_adapter else None
    )

    traffic_strategy: IDeployTrafficStrategy
    if request.traffic_strategy == "direct":
        traffic_strategy = DirectStrategy()
    else:
        assert gpu_inference_adapter is not None
        if not gpu_inference_adapter.get_deploy_status(request.model_name)["deployed"]:
            raise ValueError(
                f"{request.model_name} has no prior deploy — "
                "choose deployStrategy=direct for a model's first deploy"
            )
        traffic_strategy = BlueGreenStrategy(request.traffic_percent)

    traffic_fields = traffic_strategy.render()
    template = _JINJA_ENV.get_template("llm_inference_service.yaml.j2")
    content = template.render(
        model_name=request.model_name,
        model_version="1",
        huggingface_model_id=request.huggingface_model_id,
        serving_runtime_name=runtime_spec.serving_runtime_name,
        gpu_count=request.gpu_count,
        vllm_quantization=vllm_quantization,
        max_context_length=request.max_context_length,
        canary_traffic_percent=traffic_fields.get("canaryTrafficPercent"),
        hf_token_secret_ref=request.hf_token_secret_ref,
        batching_strategy=request.batchingStrategy,
        enable_paged_attention=request.enablePagedAttention,
        enable_prefix_caching=request.enablePrefixCaching,
        pipeline_parallel_size=request.pipelineParallelSize,
    )
    # Only "dev" + "instant" writes for real (checked above); a PR-gated
    # staging/prod manifest renders under its own environment directory so
    # the human applying it opens it in the right place.
    file_name = (
        f"infra/environments/{request.environment}/inference-services/llmops-team/"
        f"{request.model_name}/llm.yaml"
    )

    if request.release_strategy == "instant":
        assert gpu_inference_adapter is not None
        gpu_inference_adapter.deploy_llm_model(
            request.model_name,
            "1",
            request.huggingface_model_id,
            runtime_spec.serving_runtime_name,
            request.gpu_count,
            vllm_quantization,
            request.max_context_length,
            traffic_fields=traffic_fields,
            hf_token_secret_ref=request.hf_token_secret_ref,
        )
        deployed = True
    else:
        # PRGatedStrategy.release() is a no-op — safe to reuse unchanged.
        deployed = PRGatedStrategy().release(request.model_name, "1", content)["deployed"]

    # Attribute the GPU serving cost to the model's run stage. Serving is
    # recurring, so this prices a nominal 24h at the GPU's hourly rate; the
    # real gpu_type/gpu_count come straight from the form.
    gpu_hours = request.gpu_count * 24
    unit_price = gpu_hour_price(request.gpu_type)
    record_cost_event(
        stage="run",
        artifact_kind="llm-serving",
        artifact_id=request.model_name,
        version="1",
        environment=request.environment,
        cost_usd=gpu_hours * unit_price,
        quantity=float(gpu_hours),
        unit="gpu-hour",
        unit_price=unit_price,
        source="gpu-pricing",
        run_id=f"serve-{request.model_name}",
    )

    return PrepareLlmDeployResponse(file_name=file_name, content=content, deployed=deployed)
