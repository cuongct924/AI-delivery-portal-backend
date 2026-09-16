"""Tests adapters/openchoreo_inference_adapter.py.

Same kubeconfig-mocking convention as tests/test_kserve_adapter.py —
OpenChoreoInferenceAdapter.__init__ also calls
kubernetes.config.load_kube_config() eagerly.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.openchoreo_inference_adapter import (
    GROUP,
    INFERENCESERVICE_PLURAL,
    KSERVE_GROUP,
    KSERVE_VERSION,
    VERSION,
    WORKLOAD_PLURAL,
    OpenChoreoInferenceAdapter,
)


@pytest.fixture
def mock_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def adapter(mock_api: MagicMock) -> Iterator[OpenChoreoInferenceAdapter]:
    with (
        patch("adapters.openchoreo_inference_adapter.config.load_kube_config"),
        patch(
            "adapters.openchoreo_inference_adapter.client.CustomObjectsApi", return_value=mock_api
        ),
    ):
        yield OpenChoreoInferenceAdapter()


def test_deploy_model_patches_the_fixed_workload(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.patch_namespaced_custom_object.return_value = {
        "metadata": {"name": "serving-workload"}
    }

    result = adapter.deploy_model("fraud-detection-demo", "1", "models:/fraud-detection-demo/1")

    mock_api.patch_namespaced_custom_object.assert_called_once_with(
        GROUP,
        VERSION,
        "default",
        WORKLOAD_PLURAL,
        "serving-workload",
        {"spec": {"container": {"image": "models:/fraud-detection-demo/1"}}},
    )
    assert result == {"metadata": {"name": "serving-workload"}}


def test_deploy_model_raises_when_traffic_fields_given(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    with pytest.raises(NotImplementedError):
        adapter.deploy_model(
            "fraud-detection-demo",
            "1",
            "models:/fraud-detection-demo/1",
            traffic_fields={"canaryTrafficPercent": 10},
        )
    mock_api.patch_namespaced_custom_object.assert_not_called()


def test_deploy_model_ignores_empty_traffic_fields(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    adapter.deploy_model(
        "fraud-detection-demo", "1", "models:/fraud-detection-demo/1", traffic_fields={}
    )

    mock_api.patch_namespaced_custom_object.assert_called_once()


def test_get_inference_status_finds_by_label_selector(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.list_cluster_custom_object.return_value = {
        "items": [{"metadata": {"name": "serving-development-a1dd8967"}}]
    }

    result = adapter.get_inference_status("fraud-detection-demo")

    mock_api.list_cluster_custom_object.assert_called_once_with(
        KSERVE_GROUP,
        KSERVE_VERSION,
        INFERENCESERVICE_PLURAL,
        label_selector=("openchoreo.dev/component=serving,openchoreo.dev/environment=development"),
    )
    assert result == {"metadata": {"name": "serving-development-a1dd8967"}}


def test_get_inference_status_raises_404_when_reconciler_has_not_run_yet(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.list_cluster_custom_object.return_value = {"items": []}

    with pytest.raises(ApiException) as exc_info:
        adapter.get_inference_status("fraud-detection-demo")

    assert exc_info.value.status == 404


def test_predict_is_not_implemented(adapter: OpenChoreoInferenceAdapter) -> None:
    with pytest.raises(NotImplementedError):
        adapter.predict("fraud-detection-demo", {})
