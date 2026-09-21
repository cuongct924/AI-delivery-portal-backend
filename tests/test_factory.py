"""Tests adapters/factory.py's per-adapter USE_MOCK_* switches and their
USE_MOCK_ADAPTERS fallback.

Stubs the "mlflow" package at the sys.modules level before importing (same
pattern as tests/test_mlflow_adapter.py) — factory.py imports
adapters.ai_platform.mlflow_adapter at module level regardless of which branch a given
test exercises.

GpuKServeInferenceAdapter.__init__ eagerly calls load_kube_config_once(), so
the "real backend" branch of get_gpu_inference_adapter() is patched the same
way tests/test_kserve_adapter.py does.
"""

import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

import pytest  # noqa: E402

from adapters import factory  # noqa: E402
from adapters.ai_platform.mlflow_adapter import MlflowAdapter  # noqa: E402
from adapters.ai_platform.mock_misc_adapters import MockNotebookAdapter  # noqa: E402
from adapters.ai_platform.mock_model_registry_adapter import MockModelRegistryAdapter  # noqa: E402
from adapters.ai_platform.notebook_adapter import JupyterHubAdapter  # noqa: E402
from adapters.delivery.gpu_inference_adapter import GpuKServeInferenceAdapter  # noqa: E402
from adapters.delivery.mock_inference_adapter import MockInferenceAdapter  # noqa: E402
from adapters.delivery.mock_workflow_adapter import MockWorkflowAdapter  # noqa: E402
from adapters.delivery.openchoreo_inference_adapter import OpenChoreoInferenceAdapter  # noqa: E402
from adapters.delivery.openchoreo_workflow_adapter import OpenChoreoWorkflowAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every cached getter is a process-wide singleton — tests would leak
    # into each other across USE_MOCK_ADAPTERS values without this.
    factory.get_model_registry_adapter.cache_clear()
    factory.get_workflow_adapter.cache_clear()
    factory.get_notebook_adapter.cache_clear()
    factory._mock_inference_adapters.clear()
    factory._mock_gpu_inference_adapters.clear()
    # Tests below only set the specific USE_MOCK_* vars they care about and
    # assume every other one is unset — true in CI (no .env there), but not
    # on a machine whose local .env pins e.g. USE_MOCK_INFERENCE=true for
    # everyday `docker compose up` use. Start every test from a clean slate
    # instead of inheriting whatever the ambient environment happens to have.
    for var in (
        "USE_MOCK_ADAPTERS",
        "USE_MOCK_MODEL_REGISTRY",
        "USE_MOCK_WORKFLOW",
        "USE_MOCK_INFERENCE",
        "USE_MOCK_GPU_INFERENCE",
        "USE_MOCK_NOTEBOOK",
    ):
        monkeypatch.delenv(var, raising=False)


def test_model_registry_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_model_registry_adapter(), MockModelRegistryAdapter)


def test_model_registry_adapter_is_real_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    assert isinstance(factory.get_model_registry_adapter(), MlflowAdapter)


def test_workflow_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_workflow_adapter(), MockWorkflowAdapter)


def test_workflow_adapter_is_openchoreo_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Golden Paths #1/#3 migrated off Argo Server, so an unset
    # USE_MOCK_WORKFLOW resolves to the OpenChoreo backend — not ArgoAdapter.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    assert isinstance(factory.get_workflow_adapter(), OpenChoreoWorkflowAdapter)


def test_workflow_adapter_legacy_now_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # ArgoAdapter was removed in the OpenChoreo migration — a leftover
    # `USE_MOCK_WORKFLOW=legacy` must fail loudly, not silently fall back.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)
    monkeypatch.setenv("USE_MOCK_WORKFLOW", "legacy")

    with pytest.raises(ValueError, match="legacy"):
        factory.get_workflow_adapter()


def test_workflow_adapter_empty_specific_var_means_openchoreo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `.env.example` ships `USE_MOCK_WORKFLOW=` unfilled — that must land on
    # the real default, not the old legacy backend.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)
    monkeypatch.setenv("USE_MOCK_WORKFLOW", "")

    assert isinstance(factory.get_workflow_adapter(), OpenChoreoWorkflowAdapter)


def test_notebook_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_notebook_adapter(), MockNotebookAdapter)


def test_notebook_adapter_is_real_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    assert isinstance(factory.get_notebook_adapter(), JupyterHubAdapter)


def test_inference_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_inference_adapter("mlops-team"), MockInferenceAdapter)


def test_inference_adapter_is_openchoreo_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Standard model-serving migrated off KServe-direct, so an unset
    # USE_MOCK_INFERENCE resolves to the OpenChoreo backend now.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    with (
        patch("adapters.delivery.openchoreo_inference_adapter.load_kube_config_once"),
        patch("adapters.delivery.openchoreo_inference_adapter.client.CustomObjectsApi"),
    ):
        assert isinstance(factory.get_inference_adapter("mlops-team"), OpenChoreoInferenceAdapter)


def test_inference_adapter_legacy_now_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # The generic KServe-direct deploy_model() was removed from
    # adapters/delivery/gpu_inference_adapter.py (now GPU-serving-only) — a leftover
    # USE_MOCK_INFERENCE=legacy must fail loudly, not silently fall back.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)
    monkeypatch.setenv("USE_MOCK_INFERENCE", "legacy")

    with pytest.raises(ValueError, match="legacy"):
        factory.get_inference_adapter("mlops-team")


def test_inference_adapter_is_the_same_mock_instance_across_calls_for_a_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fresh instance per call would make Golden Path #2's blue-green step
    # see every model as never having a prior deploy — even one a previous
    # call in the same demo run just made.
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    first = factory.get_inference_adapter("mlops-team")
    first.deploy_model("fraud-detection", "1", "models:/fraud-detection/1")

    second = factory.get_inference_adapter("mlops-team")
    assert second is first
    second.get_inference_status("fraud-detection")  # doesn't raise 404


def test_gpu_inference_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_gpu_inference_adapter("llmops-team"), MockInferenceAdapter)


def test_gpu_inference_adapter_is_real_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    with patch("adapters.delivery.gpu_inference_adapter.load_kube_config_once"):
        assert isinstance(
            factory.get_gpu_inference_adapter("llmops-team"), GpuKServeInferenceAdapter
        )


def test_gpu_inference_adapter_mock_instance_is_separate_per_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    mlops = factory.get_gpu_inference_adapter("mlops-team")
    llmops = factory.get_gpu_inference_adapter("llmops-team")

    assert mlops is not llmops
    assert isinstance(mlops, MockInferenceAdapter)
    assert isinstance(llmops, MockInferenceAdapter)
    assert mlops.namespace == "ai-delivery-portal-dev-mlops-team"
    assert llmops.namespace == "ai-delivery-portal-dev-llmops-team"


def test_specific_flag_mocks_just_that_adapter_even_with_blanket_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model Registry/Workflow stay real, only Inference is mocked.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)
    monkeypatch.setenv("USE_MOCK_INFERENCE", "true")

    assert isinstance(factory.get_model_registry_adapter(), MlflowAdapter)
    assert isinstance(factory.get_inference_adapter("mlops-team"), MockInferenceAdapter)


def test_pure_mock_golden_path_1_does_not_500_on_latest_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproduces the exact bug report: USE_MOCK_ADAPTERS=true end to end
    # (no real cluster, no real MLflow) — trigger_workflow() -> workflow
    # reports Succeeded -> the model must already be queryable, the same
    # sequence routers/models.py's trigger_training() + get_latest_version()
    # actually run.
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    workflow_adapter = factory.get_workflow_adapter()
    workflow_adapter.trigger_workflow(
        "train-register-golden-path",
        {"model-name": "fraud-detection-demo", "task-type": "classification"},
    )

    model_registry = factory.get_model_registry_adapter()
    assert model_registry.get_latest_version("fraud-detection-demo") == "1"


def test_specific_flag_set_to_false_overrides_blanket_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")
    monkeypatch.setenv("USE_MOCK_MODEL_REGISTRY", "false")

    assert isinstance(factory.get_model_registry_adapter(), MlflowAdapter)
    assert isinstance(factory.get_workflow_adapter(), MockWorkflowAdapter)
