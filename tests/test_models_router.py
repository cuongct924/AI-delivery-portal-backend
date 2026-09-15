"""services/orchestration-api/routers/models.py — patches the module-level
`mlflow_adapter`/`argo_adapter` singleton instances and calls route functions
directly (same pattern as tests/test_prompts_router.py), no need for a
FastAPI TestClient since we're not testing the HTTP/routing layer.

Stubs the "mlflow" package at the sys.modules level before importing (same
pattern as tests/test_evaluate_drift.py and tests/test_mlflow_adapter.py) —
routers/models.py imports adapters.mlflow_adapter, which imports the real
(heavy) mlflow SDK at module level just to instantiate the module-level
MlflowAdapter() singleton, which is unnecessary for these unit tests.
"""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kubernetes.client.exceptions import ApiException

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from routers.models import (  # noqa: E402
    EnrichDatasetFeaturesRequest,
    PolicyCheckRequest,
    PrepareDeployRequest,
    RecordDeployRequest,
    RegisterModelRequest,
    TriggerTrainingRequest,
    ValidateDatasetRequest,
    enrich_dataset_features,
    get_latest_version,
    get_model_version_summary,
    get_training_status,
    list_available_features,
    list_models,
    list_recent_training_runs,
    policy_check,
    prepare_deploy_manifest,
    record_deploy,
    register_model,
    trigger_training,
    validate_dataset,
)


def test_trigger_training_sets_mode_finetune_when_base_model_uri_given() -> None:
    request = TriggerTrainingRequest(
        model_name="fraud-detection",
        dataset_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
        task_type="classification",
        algorithm="LogisticRegression",
        base_model_uri="models:/fraud-detection/1",
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-123"}}
        response = trigger_training(request)

    mock_argo.trigger_workflow.assert_called_once_with(
        "train-register-golden-path",
        {
            "model-name": "fraud-detection",
            "dataset-uri": "file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
            "task-type": "classification",
            "architecture": "sklearn",
            "algorithm": "LogisticRegression",
            "mode": "finetune",
            "base-model-uri": "models:/fraud-detection/1",
        },
    )
    assert response.workflow_name == "wf-123"


def test_trigger_training_sets_mode_train_without_base_model_uri() -> None:
    request = TriggerTrainingRequest(
        model_name="fraud-detection",
        dataset_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
        task_type="classification",
        algorithm="LogisticRegression",
        target_column="is_fraud",
        time_column="transaction_time",
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-456"}}
        response = trigger_training(request)

    mock_argo.trigger_workflow.assert_called_once_with(
        "train-register-golden-path",
        {
            "model-name": "fraud-detection",
            "dataset-uri": "file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
            "task-type": "classification",
            "architecture": "sklearn",
            "algorithm": "LogisticRegression",
            "mode": "train",
            "target-column": "is_fraud",
            "time-column": "transaction_time",
        },
    )
    assert response.workflow_name == "wf-456"


def test_trigger_training_strips_trailing_whitespace_from_dataset_uri() -> None:
    # Same class of bug as test_validate_dataset_tolerates_trailing_whitespace_
    # in_uri — an untrimmed dataset_uri would otherwise reach train.py's own
    # Path(...removeprefix("file://")) (no .strip() there either) via the
    # dataset-uri workflow parameter, so it must already be clean by the
    # time trigger_workflow() is called.
    request = TriggerTrainingRequest(
        model_name="fraud-detection",
        dataset_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv ",
        task_type="classification",
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-456"}}
        trigger_training(request)

    called_parameters = mock_argo.trigger_workflow.call_args.args[1]
    assert (
        called_parameters["dataset-uri"]
        == "file:///mnt/data/traditional-ml/fraud-detection-sample.csv"
    )


def test_trigger_training_forwards_dl_hyperparameters_for_non_sklearn_architecture() -> None:
    request = TriggerTrainingRequest(
        model_name="sensor-forecast",
        dataset_uri="file:///mnt/data/deep-learning/sensor-timeseries-sample.csv",
        task_type="regression",
        architecture="lstm",
        target_column="target",
        time_column="timestamp",
        sequence_length=10,
        num_layers=2,
        hidden_size=32,
        learning_rate=0.001,
        epochs=20,
        batch_size=16,
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-789"}}
        response = trigger_training(request)

    mock_argo.trigger_workflow.assert_called_once_with(
        "train-register-golden-path",
        {
            "model-name": "sensor-forecast",
            "dataset-uri": "file:///mnt/data/deep-learning/sensor-timeseries-sample.csv",
            "task-type": "regression",
            "architecture": "lstm",
            "mode": "train",
            "target-column": "target",
            "time-column": "timestamp",
            "sequence-length": "10",
            "num-layers": "2",
            "hidden-size": "32",
            "learning-rate": "0.001",
            "epochs": "20",
            "batch-size": "16",
        },
    )
    assert response.workflow_name == "wf-789"


def test_trigger_training_forwards_optimizer_when_set() -> None:
    request = TriggerTrainingRequest(
        model_name="sensor-forecast",
        dataset_uri="file:///mnt/data/deep-learning/sensor-timeseries-sample.csv",
        task_type="regression",
        architecture="mlp",
        target_column="target",
        hidden_layers=[8],
        dropout=0.0,
        learning_rate=0.01,
        epochs=5,
        batch_size=16,
        optimizer="sgd",
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-opt"}}
        trigger_training(request)

    call_args = mock_argo.trigger_workflow.call_args.args
    assert call_args[1]["optimizer"] == "sgd"


def test_trigger_training_forwards_byoc_fields_for_custom_algorithm() -> None:
    request = TriggerTrainingRequest(
        model_name="custom-model",
        dataset_uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
        task_type="classification",
        algorithm="custom",
        target_column="is_fraud",
        code_repo_url="https://github.com/dev/my-training-code",
        entrypoint_path="my_train.py",
        custom_config='{"lr": 0.01}',
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-byoc"}}
        response = trigger_training(request)

    mock_argo.trigger_workflow.assert_called_once_with(
        "train-register-golden-path",
        {
            "model-name": "custom-model",
            "dataset-uri": "file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
            "task-type": "classification",
            "architecture": "sklearn",
            "algorithm": "custom",
            "mode": "train",
            "target-column": "is_fraud",
            "code-repo-url": "https://github.com/dev/my-training-code",
            "entrypoint-path": "my_train.py",
            "custom-config": '{"lr": 0.01}',
        },
    )
    assert response.workflow_name == "wf-byoc"


def test_trigger_training_forwards_hpo_fields_for_non_fixed_search_strategy() -> None:
    request = TriggerTrainingRequest(
        model_name="sensor-forecast",
        dataset_uri="file:///mnt/data/deep-learning/sensor-timeseries-sample.csv",
        task_type="regression",
        architecture="mlp",
        target_column="target",
        hidden_layers=[64, 32],
        dropout=0.2,
        learning_rate=0.01,
        epochs=10,
        batch_size=16,
        search_strategy="bayesian",
        num_trials=20,
        search_space_json='{"learning_rate": {"low": 0.0001, "high": 0.1}}',
        objective_metric="r2",
        objective_direction="maximize",
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-hpo"}}
        response = trigger_training(request)

    call_args = mock_argo.trigger_workflow.call_args.args
    assert call_args[0] == "train-register-golden-path"
    assert call_args[1]["search-strategy"] == "bayesian"
    assert call_args[1]["num-trials"] == "20"
    assert call_args[1]["search-space-json"] == '{"learning_rate": {"low": 0.0001, "high": 0.1}}'
    assert call_args[1]["objective-metric"] == "r2"
    assert call_args[1]["objective-direction"] == "maximize"
    assert response.workflow_name == "wf-hpo"


def test_trigger_training_forwards_nlp_fields_for_nlp_architecture() -> None:
    request = TriggerTrainingRequest(
        model_name="review-sentiment",
        dataset_uri="file:///mnt/data/reviews-sample.csv",
        task_type="classification",
        architecture="nlp",
        target_column="sentiment",
        text_column="review",
        base_model_name="distilbert-base-uncased",
        learning_rate=5e-5,
        epochs=3,
        batch_size=16,
    )
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-nlp"}}
        response = trigger_training(request)

    call_args = mock_argo.trigger_workflow.call_args.args
    assert call_args[1]["architecture"] == "nlp"
    assert call_args[1]["text-column"] == "review"
    assert call_args[1]["base-model-name"] == "distilbert-base-uncased"
    assert call_args[1]["learning-rate"] == "5e-05"
    assert response.workflow_name == "wf-nlp"


def test_get_training_status_returns_argo_status() -> None:
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.get_workflow_status.return_value = {
            "name": "wf-123",
            "phase": "Failed",
            "message": "pod OOMKilled",
        }
        response = get_training_status("wf-123")

    mock_argo.get_workflow_status.assert_called_once_with("wf-123")
    assert response.name == "wf-123"
    assert response.phase == "Failed"
    assert response.message == "pod OOMKilled"


def test_get_training_status_records_dora_metrics_once_on_terminal_phase() -> None:
    """RQ1: a terminal-phase status must record exactly once, even though
    the frontend polls this endpoint repeatedly until it sees a terminal
    phase — see routers.models._RECORDED_TERMINAL_WORKFLOWS."""
    with (
        patch("routers.models.argo_adapter") as mock_argo,
        patch("routers.models.record_workflow_completion") as mock_record,
    ):
        mock_argo.get_workflow_status.return_value = {
            "name": "wf-dora-metrics-test",
            "phase": "Succeeded",
            "message": None,
            "started_at": "2026-09-15T01:00:00Z",
            "finished_at": "2026-09-15T01:05:00Z",
            "steps": [],
        }
        get_training_status("wf-dora-metrics-test")
        get_training_status("wf-dora-metrics-test")

    mock_record.assert_called_once_with(
        golden_path="train-track-register",
        phase="Succeeded",
        started_at="2026-09-15T01:00:00Z",
        finished_at="2026-09-15T01:05:00Z",
        steps=[],
    )


def test_get_training_status_does_not_record_metrics_while_running() -> None:
    with (
        patch("routers.models.argo_adapter") as mock_argo,
        patch("routers.models.record_workflow_completion") as mock_record,
    ):
        mock_argo.get_workflow_status.return_value = {
            "name": "wf-still-running",
            "phase": "Running",
            "message": None,
            "started_at": "2026-09-15T01:00:00Z",
            "finished_at": None,
            "steps": [],
        }
        get_training_status("wf-still-running")

    mock_record.assert_not_called()


def test_list_recent_training_runs_maps_workflow_summaries() -> None:
    with patch("routers.models.argo_adapter") as mock_argo:
        mock_argo.list_workflows.return_value = [
            {"name": "wf-123", "phase": "Succeeded", "startedAt": "2026-08-25T00:00:00Z"}
        ]
        result = list_recent_training_runs()

    assert len(result) == 1
    assert result[0].name == "wf-123"
    assert result[0].phase == "Succeeded"
    assert result[0].started_at == "2026-08-25T00:00:00Z"


def test_register_model_passes_dataset_version_through_and_tags_task_type() -> None:
    request = RegisterModelRequest(
        name="fraud-detection",
        artifact_uri="runs:/abc/model",
        task_type="classification",
        dataset_version="d41d8cd9",
    )
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.register_model.return_value = {"name": "fraud-detection", "version": "3"}
        response = register_model(request)

    mock_mlflow.register_model.assert_called_once_with(
        "fraud-detection", "runs:/abc/model", "d41d8cd9"
    )
    mock_mlflow.set_model_version_tag.assert_called_once_with(
        "fraud-detection", "3", "task_type", "classification"
    )
    assert response.name == "fraud-detection"
    assert response.version == "3"


def test_register_model_without_dataset_version_passes_none() -> None:
    request = RegisterModelRequest(
        name="fraud-detection", artifact_uri="runs:/abc/model", task_type="regression"
    )
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.register_model.return_value = {"name": "fraud-detection", "version": "1"}
        register_model(request)

    mock_mlflow.register_model.assert_called_once_with("fraud-detection", "runs:/abc/model", None)


def test_policy_check_sets_tags_and_passes_when_metrics_meet_thresholds() -> None:
    request = PolicyCheckRequest(model_name="fraud-detection", model_version="3")
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.get_model_version_details.return_value = {
            "version": "3",
            "run_id": "run-1",
            "tags": {"task_type": "classification"},
            "metrics": {"accuracy": 0.92, "precision": 0.85, "recall": 0.8, "f1": 0.82},
            "status": "READY",
        }
        result = policy_check(request)

    assert result["passed"] is True
    mock_mlflow.set_model_version_tag.assert_any_call("fraud-detection", "3", "gate_passed", "True")
    mock_mlflow.set_model_version_tag.assert_any_call(
        "fraud-detection", "3", "gate_accuracy", "0.92"
    )
    mock_mlflow.set_model_version_tag.assert_any_call(
        "fraud-detection", "3", "gate_precision", "0.85"
    )
    mock_mlflow.set_model_version_tag.assert_any_call("fraud-detection", "3", "gate_recall", "0.8")
    mock_mlflow.set_model_version_tag.assert_any_call("fraud-detection", "3", "gate_f1", "0.82")


def test_policy_check_fails_and_tags_gate_passed_false_below_threshold() -> None:
    request = PolicyCheckRequest(model_name="fraud-detection", model_version="3")
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.get_model_version_details.return_value = {
            "version": "3",
            "run_id": "run-1",
            "tags": {"task_type": "classification"},
            "metrics": {"accuracy": 0.4, "precision": 0.3, "recall": 0.3},
            "status": "READY",
        }
        result = policy_check(request)

    assert result["passed"] is False
    mock_mlflow.set_model_version_tag.assert_any_call(
        "fraud-detection", "3", "gate_passed", "False"
    )


def test_policy_check_raises_when_model_has_no_task_type_tag() -> None:
    request = PolicyCheckRequest(model_name="fraud-detection", model_version="3")
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.get_model_version_details.return_value = {
            "version": "3",
            "run_id": "run-1",
            "tags": {},
            "metrics": {"accuracy": 0.9},
            "status": "READY",
        }
        try:
            policy_check(request)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "task_type" in str(exc)


def test_validate_dataset_returns_check_results(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv_path, index=False)
    request = ValidateDatasetRequest(
        dataset_uri=f"file://{csv_path}", task_type="classification", target_column="y"
    )

    results = validate_dataset(request)

    names = {r.check_name for r in results}
    assert "check_missing_values" in names
    assert "check_duplicate_rows" in names
    assert all(r.severity in ("blocking", "warning", "info") for r in results)


def test_validate_dataset_tolerates_trailing_whitespace_in_uri(tmp_path) -> None:
    # Reproduces a real bug: a dataset_uri pasted with a trailing space (easy
    # to pick up via copy-paste from a chat/doc) made Path(...) point one
    # character past the real file, 500ing with FileNotFoundError instead of
    # a normal validation result.
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv_path, index=False)
    request = ValidateDatasetRequest(
        dataset_uri=f"file://{csv_path} ", task_type="classification", target_column="y"
    )

    results = validate_dataset(request)

    assert any(r.check_name == "check_missing_values" for r in results)


def test_enrich_dataset_features_merges_feast_features_into_dataset(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"transaction_id": [1001, 1002], "label": [0, 1]}).to_csv(csv_path, index=False)
    request = EnrichDatasetFeaturesRequest(
        dataset_uri=f"file://{csv_path}",
        entity_id_column="transaction_id",
        feature_names=["transaction_features:amount", "transaction_features:merchant_category"],
    )
    with patch("routers.models.feast_adapter") as mock_feast:
        mock_feast.get_offline_features.return_value = [
            {"entity_id": "1001", "amount": 42.5, "merchant_category": "grocery"},
            {"entity_id": "1002", "amount": 1500.0, "merchant_category": "electronics"},
        ]
        response = enrich_dataset_features(request)

    mock_feast.get_offline_features.assert_called_once_with(
        ["1001", "1002"],
        ["transaction_features:amount", "transaction_features:merchant_category"],
    )
    enriched = pd.read_csv(response.dataset_uri.removeprefix("file://"))
    assert list(enriched.columns) == ["transaction_id", "label", "amount", "merchant_category"]
    assert enriched.loc[enriched["transaction_id"] == 1002, "amount"].item() == 1500.0
    assert (
        enriched.loc[enriched["transaction_id"] == 1002, "merchant_category"].item()
        == "electronics"
    )


def test_enrich_dataset_features_overwrites_existing_column_with_feast_value(tmp_path) -> None:
    # amount already present (e.g. re-enriching) — Feast's value must win,
    # not pandas' default _x/_y suffixing on the name collision.
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"transaction_id": [1001], "amount": [0.0]}).to_csv(csv_path, index=False)
    request = EnrichDatasetFeaturesRequest(
        dataset_uri=f"file://{csv_path}",
        entity_id_column="transaction_id",
        feature_names=["transaction_features:amount"],
    )
    with patch("routers.models.feast_adapter") as mock_feast:
        mock_feast.get_offline_features.return_value = [{"entity_id": "1001", "amount": 42.5}]
        response = enrich_dataset_features(request)

    enriched = pd.read_csv(response.dataset_uri.removeprefix("file://"))
    assert list(enriched.columns) == ["transaction_id", "amount"]
    assert enriched["amount"].item() == 42.5


def test_list_available_features_returns_the_adapters_list() -> None:
    with patch("routers.models.feast_adapter") as mock_feast:
        mock_feast.list_available_features.return_value = [
            "transaction_features:amount",
            "transaction_features:merchant_category",
        ]
        response = list_available_features()

    assert response.features == [
        "transaction_features:amount",
        "transaction_features:merchant_category",
    ]


def test_list_available_features_fails_open_when_feast_errors() -> None:
    # A Feast repo that hasn't been `feast apply`-ed yet shouldn't 500 the
    # whole form — the picker just falls back to a plain text input.
    with patch("routers.models.feast_adapter") as mock_feast:
        mock_feast.list_available_features.side_effect = RuntimeError("registry not found")
        response = list_available_features()

    assert response.features == []


def test_get_model_version_summary_reads_task_type_tag() -> None:
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.get_model_version_details.return_value = {
            "version": "3",
            "run_id": "run-1",
            "tags": {"task_type": "regression"},
            "metrics": {"r2": 0.8},
            "status": "READY",
        }
        response = get_model_version_summary("house-price", "3")

    assert response.name == "house-price"
    assert response.version == "3"
    assert response.task_type == "regression"
    assert response.metrics == {"r2": 0.8}


def test_list_models_aggregates_latest_version_details() -> None:
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.list_models.return_value = [{"name": "fraud-detection"}]
        mock_mlflow.get_latest_version.return_value = "2"
        mock_mlflow.get_model_version_details.return_value = {
            "version": "2",
            "run_id": "run-1",
            "tags": {"gate_passed": "true"},
            "metrics": {"accuracy": 0.9},
            "status": "READY",
        }
        result = list_models()

    assert len(result) == 1
    assert result[0].name == "fraud-detection"
    assert result[0].version == "2"
    assert result[0].tags == {"gate_passed": "true"}
    assert result[0].metrics == {"accuracy": 0.9}


def test_list_models_skips_model_with_no_versions() -> None:
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.list_models.return_value = [{"name": "empty-model"}]
        mock_mlflow.get_latest_version.side_effect = ValueError("no registered versions")
        result = list_models()

    assert result == []
    mock_mlflow.get_model_version_details.assert_not_called()


def test_get_latest_version_returns_name_and_version() -> None:
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        mock_mlflow.get_latest_version.return_value = "5"
        response = get_latest_version("fraud-detection")

    assert response.name == "fraud-detection"
    assert response.version == "5"


def test_prepare_deploy_manifest_renders_registry_uri_into_template() -> None:
    request = PrepareDeployRequest(model_name="fraud-detection", model_version="3")

    response = prepare_deploy_manifest(request)

    assert (
        response.file_name
        == "infra/environments/dev/inference-services/mlops-team/fraud-detection/3.yaml"
    )
    assert "name: fraud-detection" in response.content
    assert 'version: "3"' in response.content
    assert "storageUri: models:/fraud-detection/3" in response.content
    assert "canaryTrafficPercent" not in response.content
    assert response.deployed is False


def test_prepare_deploy_manifest_direct_never_touches_kserve() -> None:
    # deployStrategy=direct + releaseStrategy=pr-gated (the defaults) never
    # need a kubeconfig — get_kserve_adapter() must not even be called.
    request = PrepareDeployRequest(model_name="fraud-detection", model_version="3")
    with patch("routers.models.get_kserve_adapter") as mock_get_kserve:
        prepare_deploy_manifest(request)
    mock_get_kserve.assert_not_called()


def test_prepare_deploy_manifest_traffic_split_renders_canary_percent() -> None:
    request = PrepareDeployRequest(
        model_name="fraud-detection",
        model_version="4",
        traffic_strategy="canary",
        traffic_percent=10,
    )
    with patch("routers.models.get_kserve_adapter") as mock_get_kserve:
        mock_get_kserve.return_value.get_inference_status.return_value = {"status": {}}
        response = prepare_deploy_manifest(request)

    assert "canaryTrafficPercent: 10" in response.content
    assert response.deployed is False


def test_prepare_deploy_manifest_traffic_split_without_prior_deploy_raises() -> None:
    request = PrepareDeployRequest(
        model_name="never-deployed",
        model_version="1",
        traffic_strategy="canary",
        traffic_percent=10,
    )
    with patch("routers.models.get_kserve_adapter") as mock_get_kserve:
        mock_get_kserve.return_value.get_inference_status.side_effect = ApiException(status=404)
        with pytest.raises(ValueError, match="no prior deploy"):
            prepare_deploy_manifest(request)


def test_prepare_deploy_manifest_traffic_split_requires_percent() -> None:
    request = PrepareDeployRequest(
        model_name="fraud-detection", model_version="4", traffic_strategy="canary"
    )
    with patch("routers.models.get_kserve_adapter") as mock_get_kserve:
        mock_get_kserve.return_value.get_inference_status.return_value = {"status": {}}
        with pytest.raises(ValueError, match="traffic_percent is required"):
            prepare_deploy_manifest(request)


def test_prepare_deploy_manifest_instant_deploys_without_a_pr() -> None:
    request = PrepareDeployRequest(
        model_name="fraud-detection", model_version="5", release_strategy="instant"
    )
    with patch("routers.models.get_kserve_adapter") as mock_get_kserve:
        mock_adapter = mock_get_kserve.return_value
        response = prepare_deploy_manifest(request)

    mock_adapter.deploy_model.assert_called_once_with(
        "fraud-detection", "5", "models:/fraud-detection/5", traffic_fields={}
    )
    assert response.deployed is True


def test_record_deploy_sets_deploy_pr_url_tag() -> None:
    request = RecordDeployRequest(
        model_name="fraud-detection",
        model_version="3",
        pr_url="https://github.com/org/repo/pull/1",
    )
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        response = record_deploy(request)

    mock_mlflow.set_model_version_tag.assert_called_once_with(
        "fraud-detection", "3", "deploy_pr_url", "https://github.com/org/repo/pull/1"
    )
    assert response.pr_url == "https://github.com/org/repo/pull/1"


def test_record_deploy_skips_tagging_when_no_pr_url() -> None:
    # Instant releases never open a PR — nothing to tag.
    request = RecordDeployRequest(model_name="fraud-detection", model_version="5")
    with patch("routers.models.mlflow_adapter") as mock_mlflow:
        response = record_deploy(request)

    mock_mlflow.set_model_version_tag.assert_not_called()
    assert response.pr_url is None
