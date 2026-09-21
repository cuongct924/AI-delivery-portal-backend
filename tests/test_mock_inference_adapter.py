"""Tests adapters/delivery/mock_inference_adapter.py.

Return values are typed `dict[str, object]` (they pass through the
KServe/Kubernetes API shape, per IInferenceAdapter's docstring) — cast to
`dict[str, Any]` wherever a test indexes into them more than one level deep,
same reasoning as tests/test_data_quality.py's `cast(dict, ...)`.
"""

from typing import Any, cast

import pytest
from kubernetes.client.exceptions import ApiException

from adapters.delivery.mock_inference_adapter import MockInferenceAdapter


@pytest.fixture
def adapter() -> MockInferenceAdapter:
    return MockInferenceAdapter(namespace="ai-delivery-portal-dev-mlops-team")


def test_get_inference_status_raises_404_when_never_deployed(
    adapter: MockInferenceAdapter,
) -> None:
    with pytest.raises(ApiException) as exc_info:
        adapter.get_inference_status("fraud-detection")

    assert exc_info.value.status == 404


def test_deploy_model_then_get_inference_status_reports_ready(
    adapter: MockInferenceAdapter,
) -> None:
    adapter.deploy_model("fraud-detection", "3", "models:/fraud-detection/3")

    status = cast(dict[str, Any], adapter.get_inference_status("fraud-detection"))

    assert status["status"]["conditions"] == [{"type": "Ready", "status": "True"}]
    assert status["spec"]["predictor"]["model"]["storageUri"] == "models:/fraud-detection/3"


def test_deploy_model_includes_traffic_fields(adapter: MockInferenceAdapter) -> None:
    adapter.deploy_model(
        "fraud-detection",
        "4",
        "models:/fraud-detection/4",
        traffic_fields={"canaryTrafficPercent": 10},
    )

    status = cast(dict[str, Any], adapter.get_inference_status("fraud-detection"))
    assert status["spec"]["predictor"]["canaryTrafficPercent"] == 10


def test_deploy_llm_model_then_get_inference_status_reports_ready(
    adapter: MockInferenceAdapter,
) -> None:
    adapter.deploy_llm_model("llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096)

    status = cast(dict[str, Any], adapter.get_inference_status("llama-3"))
    assert status["spec"]["predictor"]["model"]["storageUri"] == "hf://meta-llama/Llama-3-8B"


def test_predict_raises_404_when_not_deployed(adapter: MockInferenceAdapter) -> None:
    with pytest.raises(ApiException):
        adapter.predict("fraud-detection", {"input": [1, 2, 3]})


def test_predict_echoes_payload_when_deployed(adapter: MockInferenceAdapter) -> None:
    adapter.deploy_model("fraud-detection", "1", "models:/fraud-detection/1")

    result = adapter.predict("fraud-detection", {"input": [1, 2, 3]})

    assert result["echo"] == {"input": [1, 2, 3]}


def test_get_deploy_status_returns_not_deployed_when_never_deployed(
    adapter: MockInferenceAdapter,
) -> None:
    status = adapter.get_deploy_status("fraud-detection")

    assert status == {
        "deployed": False,
        "ready": False,
        "live_version": None,
        "traffic_percent": None,
    }


def test_get_deploy_status_reports_live_version_and_traffic(adapter: MockInferenceAdapter) -> None:
    adapter.deploy_model(
        "fraud-detection",
        "4",
        "models:/fraud-detection/4",
        traffic_fields={"canaryTrafficPercent": 10},
    )

    status = adapter.get_deploy_status("fraud-detection")

    assert status == {
        "deployed": True,
        "ready": True,
        "live_version": "4",
        "traffic_percent": 10,
    }


def test_get_deploy_status_works_for_a_deployed_llm(adapter: MockInferenceAdapter) -> None:
    adapter.deploy_llm_model("llama-3", "1", "meta-llama/Llama-3-8B", "vllm-runtime", 1, None, 4096)

    status = adapter.get_deploy_status("llama-3")

    assert status["deployed"] is True
    assert status["live_version"] == "1"
