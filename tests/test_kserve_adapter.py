"""Tests adapters/delivery/gpu_inference_adapter.py — the LLM self-hosted-serving golden
path's GPU/vLLM-only adapter (routers/llm_serving.py); the standard
model-serving golden path is fully on OpenChoreoInferenceAdapter now, so
this class no longer has a generic deploy_model().

GpuKServeInferenceAdapter.__init__ calls load_kube_config_once() eagerly
(needs a real kubeconfig) — patched out here, same reasoning as
test_mlflow_adapter.py patching the real mlflow SDK.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.delivery.gpu_inference_adapter import (
    GROUP,
    PLANE_LABEL_KEY,
    PLANE_LABEL_VALUE,
    PLANE_TAINT_KEY,
    PLURAL,
    VERSION,
    GpuKServeInferenceAdapter,
)


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(mock_api: MagicMock) -> Iterator[GpuKServeInferenceAdapter]:
    with (
        patch("adapters.delivery.gpu_inference_adapter.load_kube_config_once"),
        patch(
            "adapters.delivery.gpu_inference_adapter.client.CustomObjectsApi", return_value=mock_api
        ),
    ):
        yield GpuKServeInferenceAdapter(namespace="default")


def test_deploy_llm_model_omits_env_when_no_secret_ref_given(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.return_value = {"metadata": {"name": "llama-3"}}

    adapter.deploy_llm_model("llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096)

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    assert "env" not in body["spec"]["predictor"]["model"]


def test_deploy_llm_model_injects_secret_key_ref_env_var(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.return_value = {"metadata": {"name": "llama-3"}}

    adapter.deploy_llm_model(
        "llama-3",
        "1",
        "meta-llama/Llama-3.1-8B-Instruct",
        "vllm-runtime",
        1,
        None,
        4096,
        hf_token_secret_ref="llama-3-hf-token",
    )

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    env = body["spec"]["predictor"]["model"]["env"]
    assert env == [
        {
            "name": "HUGGING_FACE_HUB_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "llama-3-hf-token", "key": "token"}},
        }
    ]


def test_deploy_llm_model_pins_to_the_ai_platform_workflow_node(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.return_value = {"metadata": {"name": "llama-3"}}

    adapter.deploy_llm_model("llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096)

    predictor = mock_api.patch_namespaced_custom_object.call_args[0][-1]["spec"]["predictor"]
    assert predictor["nodeSelector"] == {PLANE_LABEL_KEY: PLANE_LABEL_VALUE}
    assert predictor["tolerations"] == [
        {
            "key": PLANE_TAINT_KEY,
            "operator": "Equal",
            "value": PLANE_LABEL_VALUE,
            "effect": "NoSchedule",
        }
    ]


def test_deploy_llm_model_creates_when_nothing_deployed_yet(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.side_effect = ApiException(status=404)
    mock_api.create_namespaced_custom_object.return_value = {"metadata": {"name": "llama-3"}}

    result = adapter.deploy_llm_model(
        "llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096
    )

    mock_api.create_namespaced_custom_object.assert_called_once()
    assert result == {"metadata": {"name": "llama-3"}}


def test_deploy_llm_model_reraises_non_404_patch_errors(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.side_effect = ApiException(status=500)

    with pytest.raises(ApiException):
        adapter.deploy_llm_model(
            "llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096
        )

    mock_api.create_namespaced_custom_object.assert_not_called()


def test_get_inference_status_passes_through(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object_status.return_value = {"status": {"conditions": []}}

    result = adapter.get_inference_status("fraud-detection")

    mock_api.get_namespaced_custom_object_status.assert_called_once_with(
        GROUP, VERSION, "default", PLURAL, "fraud-detection"
    )
    assert result == {"status": {"conditions": []}}


def test_get_deploy_status_returns_not_deployed_on_404(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object_status.side_effect = ApiException(status=404)

    status = adapter.get_deploy_status("llama-3")

    assert status == {
        "deployed": False,
        "ready": False,
        "live_version": None,
        "traffic_percent": None,
    }


def test_get_deploy_status_reports_live_version_and_traffic(
    adapter: GpuKServeInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.get_namespaced_custom_object_status.return_value = {
        "spec": {"predictor": {"canaryTrafficPercent": 20}},
        "metadata": {"labels": {"version": "4"}},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }

    status = adapter.get_deploy_status("llama-3")

    assert status == {
        "deployed": True,
        "ready": True,
        "live_version": "4",
        "traffic_percent": 20,
    }
