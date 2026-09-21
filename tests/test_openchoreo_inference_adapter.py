"""Tests adapters/delivery/openchoreo_inference_adapter.py.

Same kubeconfig-mocking convention as tests/test_kserve_adapter.py —
OpenChoreoInferenceAdapter.__init__ also calls load_kube_config_once()
eagerly.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.delivery.openchoreo_inference_adapter import (
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
        patch("adapters.delivery.openchoreo_inference_adapter.load_kube_config_once"),
        patch(
            "adapters.delivery.openchoreo_inference_adapter.client.CustomObjectsApi",
            return_value=mock_api,
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


def test_deploy_model_raises_for_a_partial_traffic_split(
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


def test_deploy_model_raises_for_a_0_percent_staged_dark_split(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    with pytest.raises(NotImplementedError):
        adapter.deploy_model(
            "fraud-detection-demo",
            "1",
            "models:/fraud-detection-demo/1",
            traffic_fields={"canaryTrafficPercent": 0},
        )
    mock_api.patch_namespaced_custom_object.assert_not_called()


def test_deploy_model_treats_a_100_percent_cutover_as_a_plain_deploy(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    # Rollback (adapters/delivery/deploy_strategies.py) always sends exactly this —
    # it must not raise, or rollback would be unusable under this backend.
    adapter.deploy_model(
        "fraud-detection-demo",
        "1",
        "models:/fraud-detection-demo/1",
        traffic_fields={"canaryTrafficPercent": 100},
    )
    mock_api.patch_namespaced_custom_object.assert_called_once()


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


def test_get_deploy_status_returns_not_deployed_when_reconciler_has_not_run_yet(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.list_cluster_custom_object.return_value = {"items": []}

    status = adapter.get_deploy_status("fraud-detection-demo")

    assert status == {
        "deployed": False,
        "ready": False,
        "live_version": None,
        "traffic_percent": None,
    }


def test_get_deploy_status_parses_version_out_of_the_models_uri_shorthand(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.list_cluster_custom_object.return_value = {
        "items": [
            {
                "spec": {
                    "predictor": {
                        "canaryTrafficPercent": 100,
                        "model": {"storageUri": "models:/fraud-detection-demo/7"},
                    }
                },
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
        ]
    }

    status = adapter.get_deploy_status("fraud-detection-demo")

    assert status == {
        "deployed": True,
        "ready": True,
        "live_version": "7",
        "traffic_percent": 100,
    }


def test_get_deploy_status_leaves_live_version_none_for_a_non_models_uri(
    adapter: OpenChoreoInferenceAdapter, mock_api: MagicMock
) -> None:
    mock_api.list_cluster_custom_object.return_value = {
        "items": [
            {
                "spec": {"predictor": {"model": {"storageUri": "s3://bucket/fraud-detection/7"}}},
                "status": {"conditions": []},
            }
        ]
    }

    status = adapter.get_deploy_status("fraud-detection-demo")

    assert status["live_version"] is None
    assert status["ready"] is False
