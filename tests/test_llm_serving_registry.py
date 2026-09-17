"""services/orchestration-api/llm_serving/registry.py."""

import pytest
from llm_serving.registry import (
    get_llm_serving_runtime,
    validate_gpu_quantization,
    validate_runtime_optimizations,
)


def test_get_llm_serving_runtime_returns_vllm_spec() -> None:
    spec = get_llm_serving_runtime("vllm")
    assert spec.serving_runtime_name == "vllm-runtime"


def test_get_llm_serving_runtime_raises_for_unknown_runtime() -> None:
    with pytest.raises(ValueError, match=r"tensorrt-llm.*triton.*vllm"):
        get_llm_serving_runtime("sglang")


def test_get_llm_serving_runtime_raises_for_roadmap_runtime() -> None:
    with pytest.raises(ValueError, match="roadmap.*only 'vLLM' runs"):
        get_llm_serving_runtime("tensorrt-llm")
    with pytest.raises(ValueError, match="roadmap.*only 'vLLM' runs"):
        get_llm_serving_runtime("triton")


def test_validate_gpu_quantization_allows_supported_combination() -> None:
    validate_gpu_quantization("H100", "fp8")  # does not raise


def test_validate_gpu_quantization_rejects_fp8_on_a100() -> None:
    with pytest.raises(ValueError, match="A100 does not support quantization='fp8'"):
        validate_gpu_quantization("A100", "fp8")


def test_validate_gpu_quantization_rejects_int8_on_b200() -> None:
    with pytest.raises(ValueError, match="B200 does not support quantization='int8'"):
        validate_gpu_quantization("B200", "int8")


def test_validate_gpu_quantization_raises_for_unknown_gpu_type() -> None:
    with pytest.raises(ValueError, match="unknown gpu_type"):
        validate_gpu_quantization("RTX4090", "none")


def test_validate_runtime_optimizations_allows_supported_flags() -> None:
    # vLLM supports these flags in MVP
    validate_runtime_optimizations(
        "vllm",
        {
            "batchingStrategy": "continuous",
            "enablePagedAttention": True,
            "enablePrefixCaching": True,
            "pipelineParallelSize": 2,
        },
    )


def test_validate_runtime_optimizations_rejects_unsupported_flag() -> None:
    # speculativeDecoding is not in vLLM's supported set for MVP
    with pytest.raises(ValueError, match="speculativeDecoding.*not supported"):
        validate_runtime_optimizations(
            "vllm",
            {
                "speculativeDecoding": "draft-model",
            },
        )


def test_validate_runtime_optimizations_allows_null_unsupported_flags() -> None:
    # None/False values for unsupported flags should be allowed (not set by user)
    validate_runtime_optimizations(
        "vllm",
        {
            "speculativeDecoding": None,
            "draftModelId": None,
        },
    )


def test_validate_runtime_optimizations_rejects_roadmap_runtime_flags() -> None:
    # tensorrt-llm has empty supported set in MVP
    with pytest.raises(ValueError, match="not supported for runtime 'tensorrt-llm'"):
        validate_runtime_optimizations(
            "tensorrt-llm",
            {
                "batchingStrategy": "continuous",
            },
        )
