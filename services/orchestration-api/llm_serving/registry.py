"""Registry of self-hosted LLM serving runtimes, plus the GPU/quantization
compatibility matrix — same registry-by-dimension pattern as
infra/argo-workflows/training-image/dl_architecture_registry.py, keyed by
runtime name instead of DL architecture.

Runtime/Optimization Support Matrix — Implemented vs Roadmap:
- vLLM: continuous batching (default), paged attention (default), prefix
  caching (configurable), speculative decoding (roadmap), disaggregation
  (roadmap), KV offload (roadmap), pipeline parallelism (configurable).
- tensorrt-llm: coming soon — runtime entry exists, but get_llm_serving_runtime
  raises a friendly error; full support in roadmap.
- triton: coming soon — same as tensorrt-llm.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class LLMServingRuntimeSpec:
    """One registry entry.

    Attributes:
        serving_runtime_name: Name of the KServe ServingRuntime CR this
            runtime maps to (see infra/llm-serving/README.md).
    """

    serving_runtime_name: str


# Runtime entries — tensorrt-llm/triton exist for UI dropdown but raise
# friendly errors at prepare time (MVP: only vLLM runs on cluster demo).
LLM_SERVING_RUNTIMES: Final[dict[str, LLMServingRuntimeSpec]] = {
    "vllm": LLMServingRuntimeSpec(serving_runtime_name="vllm-runtime"),
    "tensorrt-llm": LLMServingRuntimeSpec(serving_runtime_name="tensorrt-llm-runtime"),
    "triton": LLMServingRuntimeSpec(serving_runtime_name="triton-runtime"),
}


def get_llm_serving_runtime(runtime: str) -> LLMServingRuntimeSpec:
    """Looks up a registry entry.

    Args:
        runtime: One of the keys in LLM_SERVING_RUNTIMES.

    Returns:
        The matching LLMServingRuntimeSpec.

    Raises:
        ValueError: runtime isn't in the registry, or is in roadmap (not yet
            implemented for the cluster demo).
    """
    spec = LLM_SERVING_RUNTIMES.get(runtime)
    if spec is None:
        raise ValueError(
            f"unknown runtime {runtime!r} — must be one of {sorted(LLM_SERVING_RUNTIMES)}"
        )
    if runtime != "vllm":
        raise ValueError(
            f"Runtime '{runtime}' is on the roadmap — only 'vLLM' runs on the "
            "cluster demo today. See infra/llm-serving/README.md for the "
            "Runtime/Optimization Support Matrix."
        )
    return spec


# Per vLLM's hardware docs: A100 lacks full FP8 W8A8, B200 drops INT8.
# Enforced server-side, not just in the Scaffolder form's JSON Schema.
GPU_QUANTIZATION_COMPATIBILITY: Final[dict[str, frozenset[str]]] = {
    "L4": frozenset({"none", "fp8", "int8", "int4-awq"}),
    "L40S": frozenset({"none", "fp8", "int8", "int4-awq"}),
    "A100": frozenset({"none", "int8", "int4-awq"}),
    "H100": frozenset({"none", "fp8", "int8", "int4-awq"}),
    "H200": frozenset({"none", "fp8", "int8", "int4-awq"}),
    "B200": frozenset({"none", "fp8", "int4-awq"}),
}


# Dev-facing label -> vLLM's --quantization CLI value. "none" has no entry
# (flag omitted, auto-detected). "int8" is best-effort, unlike the
# GPU matrix above — verify against the actual model at deploy time.
VLLM_QUANTIZATION_ARGS: Final[dict[str, str]] = {
    "fp8": "fp8",
    "int8": "int8",
    "int4-awq": "awq",
}


# Optimization flags supported by each runtime — same registry-by-dimension
# pattern as GPU_QUANTIZATION_COMPATIBILITY. Only flags in this set for the
# chosen runtime pass validate_runtime_optimizations(); others raise a clear
# "not supported in MVP" error. This keeps the UI honest: the form shows all
# flags, but the backend tells you exactly which ones work today.
RUNTIME_OPTIMIZATION_SUPPORT: Final[dict[str, frozenset[str]]] = {
    "vllm": frozenset(
        {
            "batchingStrategy",
            "enablePagedAttention",
            "enablePrefixCaching",
            "pipelineParallelSize",
        }
    ),
    "tensorrt-llm": frozenset(),  # Roadmap — no flags supported in MVP
    "triton": frozenset(),  # Roadmap — no flags supported in MVP
}


def validate_gpu_quantization(gpu_type: str, quantization: str) -> None:
    """Rejects a GPU/quantization combination vLLM can't actually run.

    Args:
        gpu_type: One of GPU_QUANTIZATION_COMPATIBILITY's keys.
        quantization: The requested quantization scheme.

    Raises:
        ValueError: gpu_type is unknown, or doesn't support quantization.
    """
    supported = GPU_QUANTIZATION_COMPATIBILITY.get(gpu_type)
    if supported is None:
        raise ValueError(
            f"unknown gpu_type {gpu_type!r} — must be one of "
            f"{sorted(GPU_QUANTIZATION_COMPATIBILITY)}"
        )
    if quantization not in supported:
        raise ValueError(
            f"{gpu_type} does not support quantization={quantization!r} — "
            f"valid options: {sorted(supported)}"
        )


def validate_runtime_optimizations(runtime: str, optimizations: dict[str, object]) -> None:
    """Rejects optimization flags that the chosen runtime doesn't support.

    Args:
        runtime: One of LLM_SERVING_RUNTIMES keys.
        optimizations: Dict of optimization flag -> value from the request.

    Raises:
        ValueError: runtime is unknown, or an unsupported flag is set to a
            non-default/non-null value.
    """
    supported = RUNTIME_OPTIMIZATION_SUPPORT.get(runtime)
    if supported is None:
        raise ValueError(
            f"unknown runtime {runtime!r} — must be one of {sorted(RUNTIME_OPTIMIZATION_SUPPORT)}"
        )

    for flag, value in optimizations.items():
        if flag not in supported and value not in (None, False, "", 0):
            # Only flag an error if the value is truthy/set (not None/False/"")
            raise ValueError(
                f"Optimization '{flag}' is not supported for runtime '{runtime}' "
                f"in this MVP — supported flags for '{runtime}': {sorted(supported)}. "
                f"See infra/llm-serving/README.md for the roadmap."
            )
