"""VRAM estimation + GPU recommendation for self-hosted LLM serving —
backs the "Serve LLM (Self-hosted)" wizard's Compute & Runtime step so a
Dev picks gpuType/gpuCount from a sized recommendation instead of
guessing and hitting OOM (or over-provisioning an H100) once the pod
actually schedules.

Formula (industry rule-of-thumb, not vendor-published): weights +
activation overhead + KV cache, same shape as
https://www.hopsworks.ai/dictionary/kv-cache and vLLM's own memory
profiler output.
"""

from dataclasses import dataclass
from typing import Final

# Single-card VRAM, GB — same key set as llm_serving/registry.py's
# GPU_QUANTIZATION_COMPATIBILITY, kept as a separate table since this one
# is a hardware fact (not a vLLM software compatibility rule).
GPU_VRAM_GB: Final[dict[str, int]] = {
    "L4": 24,
    "L40S": 48,
    "A100": 80,
    "H100": 80,
    "H200": 141,
    "B200": 180,
}

# Bytes per parameter at each quantization level. "none" runs in bf16/fp16
# (2 bytes/param) — matches vLLM's default when no --quantization flag is
# passed (llm_serving/registry.py's VLLM_QUANTIZATION_ARGS has no "none"
# entry for the same reason).
_BYTES_PER_PARAM: Final[dict[str, float]] = {
    "none": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4-awq": 0.5,
}

# Every GPU count vLLM's --tensor-parallel-size accepts (registry.py's
# PrepareLlmDeployRequest.gpu_count).
_GPU_COUNTS: Final[tuple[int, ...]] = (1, 2, 4, 8)

# 20% headroom for activations/CUDA graphs/fragmentation on top of raw
# weights — a common rule of thumb (e.g. vLLM's own gpu_memory_utilization
# default leaves comparable headroom), not a per-model measurement.
_ACTIVATION_OVERHEAD_FACTOR: Final[float] = 1.2

# KV cache is stored in fp16 regardless of weight quantization — vLLM
# doesn't quantize the KV cache unless a separate --kv-cache-dtype flag is
# passed, which this Golden Path doesn't expose yet.
_KV_CACHE_BYTES_PER_ELEMENT: Final[float] = 2.0


@dataclass(frozen=True)
class GpuFitEstimate:
    """One gpuType/gpuCount combination's fit against the VRAM this
    deployment needs.

    Attributes:
        gpu_type: One of GPU_VRAM_GB's keys.
        gpu_count: Tensor-parallel degree evaluated for this fit.
        vram_available_gb: gpu_count * GPU_VRAM_GB[gpu_type].
        fits: Whether vram_available_gb covers the requested VRAM.
    """

    gpu_type: str
    gpu_count: int
    vram_available_gb: float
    fits: bool


def estimate_vram_gb(
    param_count_billion: float,
    quantization: str,
    max_context_length: int,
    concurrency: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> float:
    """Estimates the VRAM (GB) one replica needs to hold weights + KV cache.

    Args:
        param_count_billion: Model parameter count, in billions.
        quantization: One of _BYTES_PER_PARAM's keys.
        max_context_length: Max sequence length (prompt + generation).
        concurrency: Expected number of simultaneous in-flight requests
            (vLLM's --max-num-seqs equivalent) — KV cache scales linearly
            with this.
        num_layers: Model's transformer layer count (HF config.json's
            num_hidden_layers).
        hidden_size: Model's hidden dimension (HF config.json's hidden_size).
        num_attention_heads: Model's attention head count (HF config.json's
            num_attention_heads) — used only to derive head_dim.
        num_key_value_heads: Model's KV head count (HF config.json's
            num_key_value_heads — equals num_attention_heads for models
            without grouped-query attention).

    Returns:
        Estimated VRAM in GB for one replica.

    Raises:
        ValueError: quantization isn't a known scheme.
    """
    bytes_per_param = _BYTES_PER_PARAM.get(quantization)
    if bytes_per_param is None:
        raise ValueError(
            f"unknown quantization {quantization!r} — must be one of {sorted(_BYTES_PER_PARAM)}"
        )
    weights_bytes = param_count_billion * 1e9 * bytes_per_param * _ACTIVATION_OVERHEAD_FACTOR

    # 2x for K and V; head_dim derived from hidden_size/num_attention_heads
    # since HF config.json doesn't carry head_dim as its own field for most
    # architectures — GQA models only shrink num_key_value_heads, head_dim
    # stays the same as a non-GQA model of that hidden_size.
    head_dim = hidden_size / num_attention_heads
    kv_cache_per_token_bytes = (
        2 * num_layers * num_key_value_heads * head_dim * _KV_CACHE_BYTES_PER_ELEMENT
    )
    kv_cache_bytes = kv_cache_per_token_bytes * max_context_length * concurrency

    return (weights_bytes + kv_cache_bytes) / 1e9


def recommend_gpu(vram_needed_gb: float) -> list[GpuFitEstimate]:
    """Every gpuType/gpuCount combination that could serve `vram_needed_gb`,
    cheapest (fewest, smallest cards) first — the wizard shows the first
    `fits=True` entry as "Recommended" and the rest as alternatives.

    Args:
        vram_needed_gb: Output of estimate_vram_gb().

    Returns:
        One GpuFitEstimate per (gpu_type, gpu_count) pair in GPU_VRAM_GB x
        _GPU_COUNTS, sorted by (whether it fits, total VRAM ascending) so
        the smallest fitting combination sorts first.
    """
    estimates = [
        GpuFitEstimate(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            vram_available_gb=float(vram_gb * gpu_count),
            fits=vram_gb * gpu_count >= vram_needed_gb,
        )
        for gpu_type, vram_gb in GPU_VRAM_GB.items()
        for gpu_count in _GPU_COUNTS
    ]
    return sorted(estimates, key=lambda e: (not e.fits, e.vram_available_gb))
