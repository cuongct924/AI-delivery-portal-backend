"""Tests adapters/factory.py's per-adapter USE_MOCK_* switches and their
USE_MOCK_ADAPTERS fallback.

Stubs the "mlflow" package at the sys.modules level before importing (same
pattern as tests/test_mlflow_adapter.py) — factory.py imports
adapters.mlflow_adapter at module level regardless of which branch a given
test exercises.

KServeAdapter.__init__ eagerly calls kubernetes.config.load_kube_config(),
so the "real backend" branch of get_kserve_adapter() is patched the same
way tests/test_kserve_adapter.py does.
"""

import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

import pytest  # noqa: E402

from adapters import factory  # noqa: E402
from adapters.argo_adapter import ArgoAdapter  # noqa: E402
from adapters.kserve_adapter import KServeAdapter  # noqa: E402
from adapters.mlflow_adapter import MlflowAdapter  # noqa: E402
from adapters.mock_inference_adapter import MockInferenceAdapter  # noqa: E402
from adapters.mock_model_registry_adapter import MockModelRegistryAdapter  # noqa: E402
from adapters.mock_notebook_adapter import MockNotebookAdapter  # noqa: E402
from adapters.mock_workflow_adapter import MockWorkflowAdapter  # noqa: E402
from adapters.notebook_adapter import JupyterHubAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every cached getter is a process-wide singleton — tests would leak
    # into each other across USE_MOCK_ADAPTERS values without this.
    factory.get_model_registry_adapter.cache_clear()
    factory.get_workflow_adapter.cache_clear()
    factory.get_notebook_adapter.cache_clear()
    factory._mock_kserve_adapters.clear()
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


def test_workflow_adapter_is_real_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    assert isinstance(factory.get_workflow_adapter(), ArgoAdapter)


def test_notebook_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_notebook_adapter(), MockNotebookAdapter)


def test_notebook_adapter_is_real_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    assert isinstance(factory.get_notebook_adapter(), JupyterHubAdapter)


def test_kserve_adapter_is_mock_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    assert isinstance(factory.get_kserve_adapter("mlops-team"), MockInferenceAdapter)


def test_kserve_adapter_is_real_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)

    with patch("adapters.kserve_adapter.config.load_kube_config"):
        assert isinstance(factory.get_kserve_adapter("mlops-team"), KServeAdapter)


def test_kserve_adapter_is_the_same_mock_instance_across_calls_for_a_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fresh instance per call would make Golden Path #2's traffic-split/
    # canary step see every model as never having a prior deploy — even one
    # a previous call in the same demo run just made.
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    first = factory.get_kserve_adapter("mlops-team")
    first.deploy_model("fraud-detection", "1", "models:/fraud-detection/1")

    second = factory.get_kserve_adapter("mlops-team")
    assert second is first
    second.get_inference_status("fraud-detection")  # doesn't raise 404


def test_kserve_adapter_mock_instance_is_separate_per_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_MOCK_ADAPTERS", "true")

    mlops = factory.get_kserve_adapter("mlops-team")
    llmops = factory.get_kserve_adapter("llmops-team")

    assert mlops is not llmops
    assert mlops.namespace == "ai-delivery-portal-dev-mlops-team"
    assert llmops.namespace == "ai-delivery-portal-dev-llmops-team"


def test_specific_flag_mocks_just_that_adapter_even_with_blanket_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The exact combination scripts/local-demo/fake_argo.py's demo uses:
    # Model Registry/Workflow stay real, only Inference is mocked.
    monkeypatch.delenv("USE_MOCK_ADAPTERS", raising=False)
    monkeypatch.setenv("USE_MOCK_INFERENCE", "true")

    assert isinstance(factory.get_model_registry_adapter(), MlflowAdapter)
    assert isinstance(factory.get_kserve_adapter("mlops-team"), MockInferenceAdapter)


def test_pure_mock_golden_path_1_does_not_500_on_latest_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproduces the exact bug report: USE_MOCK_ADAPTERS=true end to end
    # (no fake_argo.py, no real MLflow) — trigger_workflow() -> workflow
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
