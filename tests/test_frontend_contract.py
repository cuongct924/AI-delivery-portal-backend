"""Consumer-driven contract guard between the Portal frontend and this API.

The Portal (separate repo, `cuongct924/AI-delivery-portal-frontend`) is the
only caller of these golden-path endpoints. Its Scaffolder actions build a
JSON body per action and POST it here. Pydantic's default `extra="ignore"`
means a field the frontend sends but the backend doesn't declare is dropped
*silently* — the request still returns 200, so the drift only shows up as a
feature that quietly does nothing (or, when a backend field is required and
the frontend omits it, a runtime 422 in the Portal).

Each entry below is the exact body the corresponding frontend action sends
(see `packages/backend/src/actions/*.ts`). The test asserts two things:

1. the backend request model accepts the payload, and
2. every key the frontend sends maps to a declared model field — no silent
   drops.

When the frontend adds/renames a field, update the payload here; the test
then fails until the backend model catches up.
"""

import pytest
from pydantic import BaseModel
from routers.costs import CostCheckRequest, EstimateCostRequest
from routers.eval_sets import DraftEvalSetRequest
from routers.llm_serving import PrepareLlmDeployRequest
from routers.models import (
    ConfirmPromotionRequest,
    EnrichDatasetFeaturesRequest,
    PolicyCheckRequest,
    PrepareDeployRequest,
    PromoteRequest,
    RecordDeployRequest,
    RegisterModelRequest,
    RollbackPromotionRequest,
    TriggerTrainingRequest,
    ValidateDatasetRequest,
)
from routers.monitoring import SetupMonitoringRequest
from routers.notebooks import CreateNotebookRequest
from routers.prompts import ActivatePromptRequest, DraftPromptRequest, EvaluatePromptRequest
from routers.rag import RagActivateRequest, RagEvaluateRequest, RagIngestRequest

# (action id, request model, exact body the frontend action sends)
CONTRACTS: list[tuple[str, type[BaseModel], dict[str, object]]] = [
    (
        "orchestration:trigger-training",
        TriggerTrainingRequest,
        {
            "model_name": "fraud-detection",
            "dataset_uri": "file:///mnt/data/telco-fraud-detection-sample.csv",
            "task_type": "classification",
            "architecture": "sklearn",
            "algorithm": "LogisticRegression",
            "target_column": "is_fraud",
            "id_columns": ["customer_id"],
            "time_column": "event_time",
            "base_model_uri": "models:/fraud-detection/1",
            "hidden_layers": [64, 32],
            "dropout": 0.2,
            "sequence_length": 128,
            "num_layers": 4,
            "hidden_size": 256,
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 32,
            "optimizer": "adam",
            "search_strategy": "random",
            "num_trials": 5,
            "search_space_json": "{}",
            "objective_metric": "f1",
            "objective_direction": "maximize",
            "text_column": "text",
            "base_model_name": "distilbert-base-uncased",
        },
    ),
    (
        "orchestration:validate-dataset",
        ValidateDatasetRequest,
        {
            "dataset_uri": "file:///mnt/data/telco-fraud-detection-sample.csv",
            "task_type": "classification",
            "target_column": "is_fraud",
            "time_column": "event_time",
        },
    ),
    (
        "orchestration:enrich-dataset-features",
        EnrichDatasetFeaturesRequest,
        {
            "dataset_uri": "file:///mnt/data/telco-fraud-detection-sample.csv",
            "entity_id_column": "customer_id",
            "feature_names": ["transaction_features:amount"],
        },
    ),
    (
        "orchestration:register-model",
        RegisterModelRequest,
        {
            "name": "fraud-detection",
            "artifact_uri": "runs:/abc123/model",
            "task_type": "classification",
            "dataset_version": "v1",
        },
    ),
    (
        "orchestration:policy-check",
        PolicyCheckRequest,
        {"model_name": "fraud-detection", "model_version": "3"},
    ),
    (
        "orchestration:estimate-cost",
        EstimateCostRequest,
        {
            "golden_path": "train-track-register",
            "stage": "train",
            "artifact": "fraud-detection",
            "params": {"epochs": 10},
        },
    ),
    (
        "orchestration:cost-gate",
        CostCheckRequest,
        {
            "golden_path": "train-track-register",
            "stage": "train",
            "artifact": "fraud-detection",
            "params": {"epochs": 10},
            "mode": "warn",
        },
    ),
    (
        "orchestration:prepare-deploy-manifest",
        PrepareDeployRequest,
        {
            "model_name": "fraud-detection",
            "model_version": "3",
            "traffic_strategy": "canary",
            "traffic_percent": 10,
            "release_strategy": "instant",
            "action": "deploy",
            "enable_prediction_logging": True,
        },
    ),
    (
        "orchestration:record-deploy",
        RecordDeployRequest,
        {
            "model_name": "fraud-detection",
            "model_version": "3",
            "pr_url": "https://github.com/org/repo/pull/1",
        },
    ),
    (
        "orchestration:promote-model",
        PromoteRequest,
        {"target_environment": "staging"},
    ),
    (
        "orchestration:rollback-promotion",
        RollbackPromotionRequest,
        {"environment": "staging"},
    ),
    (
        "orchestration:confirm-promotion",
        ConfirmPromotionRequest,
        {
            "environment": "staging",
            "project_release": "fraud-detection-staging",
            "event_type": "deploy",
        },
    ),
    (
        "orchestration:setup-monitoring",
        SetupMonitoringRequest,
        {
            "model_name": "fraud-detection",
            "model_version": "3",
            "reference_data_uri": "file:///mnt/data/telco-fraud-detection-sample.csv",
            "production_data_source": "managed-prediction-log",
            "production_data_uri": "file:///mnt/monitoring/fraud-detection-recent.csv",
            "schedule": "0 0 * * *",
            "monitoring_type": "performance-degradation",
            "drift_threshold": 0.5,
            "ground_truth_data_uri": "file:///mnt/monitoring/fraud-detection-labels.csv",
            "ground_truth_data_source": "managed-label-log",
            "metric_name": "f1_score",
            "min_metric_threshold": 0.85,
            "on_drift_detected": "alert-only",
            "retrain_request_json": '{"model_name": "fraud-detection"}',
            "failure_webhook_url": "http://portal.test/api/monitoring/failures",
        },
    ),
    (
        "orchestration:prepare-llm-deploy-manifest",
        PrepareLlmDeployRequest,
        {
            "model_name": "qwen-7b",
            "huggingface_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "runtime": "vllm",
            "gpu_type": "a100",
            "gpu_count": 1,
            "quantization": "none",
            "max_context_length": 4096,
            "traffic_strategy": "direct",
            "traffic_percent": 100,
            "release_strategy": "instant",
            "environment": "dev",
            "hf_token_secret_ref": "hf-token",
        },
    ),
    (
        "orchestration:rag-ingest",
        RagIngestRequest,
        {
            "collection": "product-docs",
            "source_paths": ["s3://bucket/docs/"],
            "chunk_size": 800,
            "chunk_overlap": 100,
        },
    ),
    (
        "orchestration:rag-evaluate",
        RagEvaluateRequest,
        {
            "collection": "product-docs",
            "index_version": "1",
            "eval_cases": [{"question": "What is the refund policy?"}],
            "model": "claude-sonnet-5",
        },
    ),
    (
        "orchestration:rag-activate",
        RagActivateRequest,
        {
            "collection": "product-docs",
            "index_version": "1",
            "environment": "production",
            "is_rollback": False,
        },
    ),
    (
        "orchestration:draft-prompt",
        DraftPromptRequest,
        {"name": "support-agent", "persona": "assistant", "content": "You are helpful."},
    ),
    (
        "orchestration:evaluate-prompt",
        EvaluatePromptRequest,
        {
            "version": "1",
            "eval_cases": [{"question": "What is the refund policy?"}],
            "model": "claude-sonnet-5",
        },
    ),
    (
        "orchestration:activate-prompt",
        ActivatePromptRequest,
        {"version": "1", "environment": "production", "is_rollback": False},
    ),
    (
        "orchestration:draft-eval-set",
        DraftEvalSetRequest,
        {"name": "support-eval", "questions": ["What is the refund policy?"]},
    ),
    (
        "orchestration:create-notebook",
        CreateNotebookRequest,
        {
            "environment": "pytorch-cuda",
            "cpu_cores": 2,
            "ram_gb": 8,
            "gpu_type": "t4",
            "gpu_count": 1,
            "storage_gb": 20,
            "idle_timeout_minutes": 60,
        },
    ),
]


@pytest.mark.parametrize(
    ("action_id", "model", "payload"),
    CONTRACTS,
    ids=[action_id for action_id, _, _ in CONTRACTS],
)
def test_frontend_payload_matches_backend_model(
    action_id: str, model: type[BaseModel], payload: dict[str, object]
) -> None:
    del action_id
    model(**payload)

    dropped = set(payload) - set(model.model_fields)
    assert not dropped, (
        f"{model.__name__} silently drops frontend field(s) {sorted(dropped)} — "
        "add them to the request model (or remove them from the frontend action)"
    )
