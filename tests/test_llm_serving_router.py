"""services/orchestration-api/routers/llm_serving.py — patches
`routers.llm_serving.get_gpu_inference_adapter`/`routers.llm_serving.huggingface_hub_adapter`
and calls the route function directly, same pattern as
tests/test_models_router.py's prepare_deploy_manifest tests. No mlflow
stub needed — this router doesn't import the mlflow SDK.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from routers.llm_serving import (
    PrepareLlmDeployRequest,
    get_gpu_recommendation,
    get_rollout_eligibility,
    prepare_llm_deploy_manifest,
    validate_huggingface_model,
)

_ADMIN_USER = {"sub": "test-user", "roles": ["llm-ops-admin"]}
_NON_ADMIN_USER = {"sub": "test-user", "roles": []}


def test_prepare_llm_deploy_manifest_renders_manifest_correctly() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
    )

    response = prepare_llm_deploy_manifest(request)

    assert (
        response.file_name
        == "infra/environments/dev/inference-services/llmops-team/llama-3-8b/llm.yaml"
    )
    assert "name: llama-3-8b" in response.content
    assert 'version: "1"' in response.content
    assert "storageUri: hf://meta-llama/Llama-3.1-8B-Instruct" in response.content
    assert "--tensor-parallel-size=1" in response.content
    assert "--max-model-len=4096" in response.content
    assert "--quantization" not in response.content  # quantization="none" default
    assert "canaryTrafficPercent" not in response.content
    assert response.deployed is False


def test_prepare_llm_deploy_manifest_direct_never_touches_the_gpu_adapter() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
    )
    with patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu:
        prepare_llm_deploy_manifest(request)
    mock_get_gpu.assert_not_called()


def test_prepare_llm_deploy_manifest_renders_quantization_arg() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        quantization="fp8",
    )

    response = prepare_llm_deploy_manifest(request)

    assert "--quantization=fp8" in response.content


def test_prepare_llm_deploy_manifest_omits_speculative_model_when_none() -> None:
    # The frontend always sends speculativeDecoding="none" + a default
    # draftModelId — neither must render an arg, and neither may 500.
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        speculativeDecoding="none",
        draftModelId="meta-llama/Llama-3.2-1B-Instruct",
    )

    response = prepare_llm_deploy_manifest(request)

    assert "--speculative-model" not in response.content


def test_prepare_llm_deploy_manifest_renders_ngram_speculative_model() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        speculativeDecoding="ngram",
    )

    response = prepare_llm_deploy_manifest(request)

    assert "--speculative-model=[ngram]" in response.content


def test_prepare_llm_deploy_manifest_renders_draft_model_speculative_model() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        speculativeDecoding="draft-model",
        draftModelId="meta-llama/Llama-3.2-1B-Instruct",
    )

    response = prepare_llm_deploy_manifest(request)

    assert "--speculative-model=meta-llama/Llama-3.2-1B-Instruct" in response.content


def test_prepare_llm_deploy_manifest_draft_model_without_draft_id_raises() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        speculativeDecoding="draft-model",
    )

    with pytest.raises(ValueError, match="requires draftModelId"):
        prepare_llm_deploy_manifest(request)


def test_prepare_llm_deploy_manifest_blue_green_renders_a_100_percent_cutover() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        traffic_strategy="blue-green",
    )
    with patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu:
        mock_get_gpu.return_value.get_deploy_status.return_value = {
            "deployed": True,
            "ready": True,
            "live_version": "1",
            "traffic_percent": 100,
        }
        response = prepare_llm_deploy_manifest(request)

    assert "canaryTrafficPercent: 100" in response.content
    assert response.deployed is False


def test_prepare_llm_deploy_manifest_blue_green_without_prior_deploy_raises() -> None:
    request = PrepareLlmDeployRequest(
        model_name="never-deployed",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        traffic_strategy="blue-green",
    )
    with patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu:
        mock_get_gpu.return_value.get_deploy_status.return_value = {
            "deployed": False,
            "ready": False,
            "live_version": None,
            "traffic_percent": None,
        }
        with pytest.raises(ValueError, match="no prior deploy"):
            prepare_llm_deploy_manifest(request)


def test_prepare_llm_deploy_manifest_instant_calls_deploy_llm_model() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        gpu_count=2,
        quantization="fp8",
        release_strategy="instant",
    )
    with patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu:
        response = prepare_llm_deploy_manifest(request, user=_ADMIN_USER)

    mock_get_gpu.return_value.deploy_llm_model.assert_called_once_with(
        "llama-3-8b",
        "1",
        "meta-llama/Llama-3.1-8B-Instruct",
        "vllm-runtime",
        2,
        "fp8",
        4096,
        traffic_fields={},
        hf_token_secret_ref=None,
    )
    assert response.deployed is True


def test_prepare_llm_deploy_manifest_instant_requires_admin_role() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        release_strategy="instant",
    )
    with (
        patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu,
        pytest.raises(HTTPException) as exc_info,
    ):
        prepare_llm_deploy_manifest(request, user=_NON_ADMIN_USER)
    assert exc_info.value.status_code == 403
    mock_get_gpu.return_value.deploy_llm_model.assert_not_called()


def test_prepare_llm_deploy_manifest_instant_rejects_non_dev_environment() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        release_strategy="instant",
        environment="staging",
    )
    with (
        patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu,
        pytest.raises(ValueError, match="only possible for environment='dev'"),
    ):
        prepare_llm_deploy_manifest(request, user=_ADMIN_USER)
    mock_get_gpu.assert_not_called()


def test_prepare_llm_deploy_manifest_rejects_unknown_environment() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        environment="canary-env",
    )
    with pytest.raises(ValueError, match="unknown environment"):
        prepare_llm_deploy_manifest(request)


def test_prepare_llm_deploy_manifest_pr_gated_staging_renders_under_environment_path() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        environment="staging",
    )
    response = prepare_llm_deploy_manifest(request)

    assert (
        response.file_name
        == "infra/environments/staging/inference-services/llmops-team/llama-3-8b/llm.yaml"
    )
    assert response.deployed is False


def test_prepare_llm_deploy_manifest_renders_hf_token_secret_ref() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="H100",
        hf_token_secret_ref="llama-3-hf-token",
    )
    response = prepare_llm_deploy_manifest(request)

    assert "name: llama-3-hf-token" in response.content
    assert "HUGGING_FACE_HUB_TOKEN" in response.content


def test_prepare_llm_deploy_manifest_rejects_incompatible_gpu_quantization() -> None:
    request = PrepareLlmDeployRequest(
        model_name="llama-3-8b",
        huggingface_model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_type="A100",
        quantization="fp8",
    )
    with (
        patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu,
        pytest.raises(ValueError, match="A100 does not support"),
    ):
        prepare_llm_deploy_manifest(request)
    mock_get_gpu.assert_not_called()


def test_validate_huggingface_model_returns_adapter_info() -> None:
    with patch("routers.llm_serving.huggingface_hub_adapter") as mock_adapter:
        mock_adapter.get_model_info.return_value = {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "exists": True,
            "is_gated": False,
            "param_count_billion": 7.25,
            "max_context_length": 32768,
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "license": "apache-2.0",
        }
        response = validate_huggingface_model("mistralai/Mistral-7B-Instruct-v0.3")

    mock_adapter.get_model_info.assert_called_once_with("mistralai/Mistral-7B-Instruct-v0.3")
    assert response.exists is True
    assert response.param_count_billion == 7.25


def test_search_huggingface_models_returns_adapter_ids() -> None:
    from routers.llm_serving import search_huggingface_models

    with patch("routers.llm_serving.huggingface_hub_adapter") as mock_adapter:
        mock_adapter.search_models.return_value = ["a/b"]
        response = search_huggingface_models(q="a", limit=20)

    mock_adapter.search_models.assert_called_once_with("a", 20)
    assert response.model_ids == ["a/b"]


def test_list_secrets_returns_sorted_names() -> None:
    from types import SimpleNamespace

    from routers.llm_serving import list_secrets

    items = [
        SimpleNamespace(metadata=SimpleNamespace(name="b")),
        SimpleNamespace(metadata=SimpleNamespace(name="a")),
    ]
    with (
        patch("adapters.delivery._kube_client.load_kube_config_once"),
        patch("kubernetes.client.CoreV1Api") as mock_core,
    ):
        mock_core.return_value.list_namespaced_secret.return_value = SimpleNamespace(items=items)
        response = list_secrets(namespace="default")

    assert response.names == ["a", "b"]


def test_list_secrets_falls_back_to_empty_without_a_cluster() -> None:
    from routers.llm_serving import list_secrets

    with (
        patch(
            "adapters.delivery._kube_client.load_kube_config_once",
            side_effect=Exception("nope"),
        ),
        patch("kubernetes.client.CoreV1Api"),
    ):
        response = list_secrets(namespace="default")

    assert response.names == []


def test_get_gpu_recommendation_returns_smallest_fitting_gpu() -> None:
    response = get_gpu_recommendation(
        param_count_billion=8.0,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    assert response.recommended is not None
    assert response.recommended.fits is True
    assert len(response.estimates) == 24


def test_get_rollout_eligibility_true_when_prior_deploy_exists() -> None:
    with patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu:
        mock_get_gpu.return_value.get_deploy_status.return_value = {
            "deployed": True,
            "ready": True,
            "live_version": "1",
            "traffic_percent": None,
        }
        response = get_rollout_eligibility("llama-3-8b")
    assert response.has_prior_deploy is True


def test_get_rollout_eligibility_false_when_no_prior_deploy() -> None:
    with patch("routers.llm_serving.get_gpu_inference_adapter") as mock_get_gpu:
        mock_get_gpu.return_value.get_deploy_status.return_value = {
            "deployed": False,
            "ready": False,
            "live_version": None,
            "traffic_percent": None,
        }
        response = get_rollout_eligibility("never-deployed")
    assert response.has_prior_deploy is False
