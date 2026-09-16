"""Tests adapters/kserve_adapter.py.

KServeAdapter.__init__ calls kubernetes.config.load_kube_config() eagerly
(needs a real kubeconfig) — patched out here, same reasoning as
test_mlflow_adapter.py patching the real mlflow SDK.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.kserve_adapter import GROUP, PLURAL, VERSION, KServeAdapter


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(mock_api: MagicMock) -> Iterator[KServeAdapter]:
    with (
        patch("adapters.kserve_adapter.config.load_kube_config"),
        patch("adapters.kserve_adapter.client.CustomObjectsApi", return_value=mock_api),
    ):
        yield KServeAdapter(namespace="default")


def test_deploy_model_patches_when_already_deployed(
    adapter: KServeAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.return_value = {"metadata": {"name": "fraud-detection"}}

    result = adapter.deploy_model("fraud-detection", "3", "models:/fraud-detection/3")

    mock_api.patch_namespaced_custom_object.assert_called_once()
    mock_api.create_namespaced_custom_object.assert_not_called()
    assert result == {"metadata": {"name": "fraud-detection"}}


def test_deploy_model_creates_when_nothing_deployed_yet(
    adapter: KServeAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.side_effect = ApiException(status=404)
    mock_api.create_namespaced_custom_object.return_value = {
        "metadata": {"name": "fraud-detection"}
    }

    result = adapter.deploy_model("fraud-detection", "1", "models:/fraud-detection/1")

    mock_api.create_namespaced_custom_object.assert_called_once()
    assert result == {"metadata": {"name": "fraud-detection"}}


def test_deploy_model_reraises_non_404_patch_errors(
    adapter: KServeAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.side_effect = ApiException(status=500)

    with pytest.raises(ApiException):
        adapter.deploy_model("fraud-detection", "3", "models:/fraud-detection/3")

    mock_api.create_namespaced_custom_object.assert_not_called()


def test_deploy_model_includes_traffic_fields_in_predictor_spec(
    adapter: KServeAdapter, mock_api: MagicMock
) -> None:
    adapter.deploy_model(
        "fraud-detection",
        "4",
        "models:/fraud-detection/4",
        traffic_fields={"canaryTrafficPercent": 10},
    )

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    assert body["spec"]["predictor"]["canaryTrafficPercent"] == 10
    assert body["spec"]["predictor"]["model"]["storageUri"] == "models:/fraud-detection/4"


def test_deploy_model_omits_traffic_fields_when_none_given(
    adapter: KServeAdapter, mock_api: MagicMock
) -> None:
    adapter.deploy_model("fraud-detection", "3", "models:/fraud-detection/3")

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    assert "canaryTrafficPercent" not in body["spec"]["predictor"]


def test_deploy_llm_model_omits_env_when_no_secret_ref_given(
    adapter: KServeAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.return_value = {"metadata": {"name": "llama-3"}}

    adapter.deploy_llm_model("llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096)

    body = mock_api.patch_namespaced_custom_object.call_args[0][-1]
    assert "env" not in body["spec"]["predictor"]["model"]


def test_deploy_llm_model_injects_secret_key_ref_env_var(
    adapter: KServeAdapter, mock_api: MagicMock
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


def test_get_inference_status_passes_through(adapter: KServeAdapter, mock_api: MagicMock) -> None:
    mock_api.get_namespaced_custom_object_status.return_value = {"status": {"conditions": []}}

    result = adapter.get_inference_status("fraud-detection")

    mock_api.get_namespaced_custom_object_status.assert_called_once_with(
        GROUP, VERSION, "default", PLURAL, "fraud-detection"
    )
    assert result == {"status": {"conditions": []}}
