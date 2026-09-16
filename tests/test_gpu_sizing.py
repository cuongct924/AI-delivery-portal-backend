"""services/orchestration-api/llm_serving/gpu_sizing.py."""

import pytest
from llm_serving.gpu_sizing import estimate_vram_gb, recommend_gpu

# Llama-3-8B-Instruct's real published architecture (also
# adapters/mock_huggingface_hub_adapter.py's _KNOWN_MODELS entry).
_LLAMA_3_8B_ARCHITECTURE = {
    "num_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}


def test_estimate_vram_gb_matches_known_8b_ballpark() -> None:
    vram_gb = estimate_vram_gb(
        param_count_billion=8.0,
        quantization="none",
        max_context_length=4096,
        concurrency=1,
        **_LLAMA_3_8B_ARCHITECTURE,
    )
    # ~16GB raw fp16 weights * 1.2 headroom + a few hundred MB of KV cache
    # — a real 8B model comfortably fits a single 24GB card, which this
    # range confirms without hard-coding the exact float.
    assert 18.0 < vram_gb < 21.0


def test_estimate_vram_gb_scales_with_concurrency() -> None:
    low = estimate_vram_gb(
        param_count_billion=8.0,
        quantization="none",
        max_context_length=4096,
        concurrency=1,
        **_LLAMA_3_8B_ARCHITECTURE,
    )
    high = estimate_vram_gb(
        param_count_billion=8.0,
        quantization="none",
        max_context_length=4096,
        concurrency=32,
        **_LLAMA_3_8B_ARCHITECTURE,
    )
    assert high > low


def test_estimate_vram_gb_quantization_reduces_weight_footprint() -> None:
    unquantized = estimate_vram_gb(
        param_count_billion=70.0,
        quantization="none",
        max_context_length=4096,
        concurrency=1,
        **_LLAMA_3_8B_ARCHITECTURE,
    )
    quantized = estimate_vram_gb(
        param_count_billion=70.0,
        quantization="int4-awq",
        max_context_length=4096,
        concurrency=1,
        **_LLAMA_3_8B_ARCHITECTURE,
    )
    assert quantized < unquantized


def test_estimate_vram_gb_rejects_unknown_quantization() -> None:
    with pytest.raises(ValueError, match="unknown quantization"):
        estimate_vram_gb(
            param_count_billion=8.0,
            quantization="int2",
            max_context_length=4096,
            concurrency=1,
            **_LLAMA_3_8B_ARCHITECTURE,
        )


def test_recommend_gpu_recommends_smallest_fitting_combination() -> None:
    estimates = recommend_gpu(vram_needed_gb=20.0)
    first_fitting = next(e for e in estimates if e.fits)
    # L4 (24GB) x1 is the cheapest combination that covers 20GB.
    assert first_fitting.gpu_type == "L4"
    assert first_fitting.gpu_count == 1


def test_recommend_gpu_scales_up_gpu_count_for_a_large_model() -> None:
    estimates = recommend_gpu(vram_needed_gb=300.0)
    first_fitting = next(e for e in estimates if e.fits)
    assert first_fitting.vram_available_gb >= 300.0


def test_recommend_gpu_every_combination_present() -> None:
    estimates = recommend_gpu(vram_needed_gb=1.0)
    # 6 GPU types x 4 gpu_count options (llm_serving/gpu_sizing.py's
    # GPU_VRAM_GB x _GPU_COUNTS).
    assert len(estimates) == 24
