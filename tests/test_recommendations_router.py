"""services/orchestration-api/routers/recommendations.py — same pattern as
tests/test_models_router.py: patches the module-level `argo_adapter`
singleton and calls route functions directly."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.modules.setdefault("mlflow", MagicMock())
sys.modules.setdefault("mlflow.tracking", MagicMock())

from routers.recommendations import (  # noqa: E402
    TriggerRecTrainingRequest,
    ValidateRecDatasetRequest,
    trigger_rec_training,
    validate_rec_dataset,
)


def test_trigger_rec_training_forwards_required_fields() -> None:
    request = TriggerRecTrainingRequest(
        model_name="product-recommender",
        interactions_uri="file:///mnt/data/recsys/interactions-sample.csv",
        user_id_column="user_id",
        item_id_column="item_id",
        timestamp_column="timestamp",
        algorithm="als",
    )
    with patch("routers.recommendations.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-rec-1"}}
        response = trigger_rec_training(request)

    mock_argo.trigger_workflow.assert_called_once_with(
        "rec-train-register-golden-path",
        {
            "model-name": "product-recommender",
            "interactions-uri": "file:///mnt/data/recsys/interactions-sample.csv",
            "user-id-column": "user_id",
            "item-id-column": "item_id",
            "timestamp-column": "timestamp",
            "algorithm": "als",
            "k": "10",
        },
    )
    assert response.workflow_name == "wf-rec-1"


def test_trigger_rec_training_forwards_optional_fields_when_set() -> None:
    request = TriggerRecTrainingRequest(
        model_name="content-recommender",
        interactions_uri="file:///mnt/data/recsys/interactions-sample.csv",
        user_id_column="user_id",
        item_id_column="item_id",
        timestamp_column="timestamp",
        algorithm="tfidf_cosine",
        k=5,
        hyperparameters_json="{}",
        item_features_uri="file:///mnt/data/recsys/item-features-sample.csv",
        item_id_column_features="item_id",
        item_text_column="description",
    )
    with patch("routers.recommendations.argo_adapter") as mock_argo:
        mock_argo.trigger_workflow.return_value = {"metadata": {"name": "wf-rec-2"}}
        trigger_rec_training(request)

    call_args = mock_argo.trigger_workflow.call_args.args
    assert call_args[1]["item-features-uri"] == "file:///mnt/data/recsys/item-features-sample.csv"
    assert call_args[1]["item-id-column-features"] == "item_id"
    assert call_args[1]["item-text-column"] == "description"
    assert call_args[1]["k"] == "5"


def test_validate_rec_dataset_runs_checks_against_the_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1"],
            "item_id": ["i1", "i2", "i1"],
            "timestamp": [1, 2, 3],
        }
    ).to_csv(csv_path, index=False)
    request = ValidateRecDatasetRequest(
        interactions_uri=f"file://{csv_path}",
        user_id_column="user_id",
        item_id_column="item_id",
    )

    results = validate_rec_dataset(request)

    results_by_name = {r.check_name: r for r in results}
    assert "check_rec_duplicate_interactions" in results_by_name
    # 1 duplicate (user_id, item_id) pair: (u1, i1).
    assert results_by_name["check_rec_duplicate_interactions"].severity == "warning"
