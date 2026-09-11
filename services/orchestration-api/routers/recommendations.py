"""Recommendation System API — Golden Path #3. A separate router from
models.py (Golden Path #1/#2): the dataset contract (multi-file manifest)
and the training trigger's shape are different enough that reusing
TriggerTrainingRequest/validate_dataset would mean optional fields nobody
else uses tacked onto an unrelated model. `/models/register`,
`/policy-check`, `/deploy-model/*` (models.py) are reused unchanged —
register→gate→deploy doesn't care which Golden Path produced the model.
"""

from pathlib import Path
from typing import Final, cast

import pandas as pd
from auth.thunder import get_current_user
from data_quality.checks import CheckResult
from data_quality.registry import run_rec_checks
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from adapters.factory import get_workflow_adapter

router = APIRouter(tags=["recommendations"])

argo_adapter = get_workflow_adapter()

REC_TRAIN_REGISTER_TEMPLATE: Final[str] = "rec-train-register-golden-path"


class TriggerRecTrainingRequest(BaseModel):
    model_name: str
    interactions_uri: str
    user_id_column: str
    item_id_column: str
    timestamp_column: str
    algorithm: str
    k: int = 10
    hyperparameters_json: str | None = None
    # Required for algorithm="svd"/"knn" (collaborative_explicit).
    rating_column: str | None = None
    # Both required together for algorithm="tfidf_cosine" (content_based).
    item_features_uri: str | None = None
    item_id_column_features: str | None = None
    item_text_column: str | None = None


class TriggerRecTrainingResponse(BaseModel):
    workflow_name: str


class ValidateRecDatasetRequest(BaseModel):
    interactions_uri: str
    user_id_column: str
    item_id_column: str


class CheckResultResponse(BaseModel):
    check_name: str
    severity: str
    message: str
    details: dict[str, object]

    @classmethod
    def from_check_result(cls, result: CheckResult) -> "CheckResultResponse":
        return cls(
            check_name=result.check_name,
            severity=result.severity,
            message=result.message,
            details=result.details,
        )


@router.post("/trigger-rec-training", response_model=TriggerRecTrainingResponse)
def trigger_rec_training(
    request: TriggerRecTrainingRequest, user: dict = Depends(get_current_user)
) -> TriggerRecTrainingResponse:
    parameters = {
        "model-name": request.model_name,
        "interactions-uri": request.interactions_uri,
        "user-id-column": request.user_id_column,
        "item-id-column": request.item_id_column,
        "timestamp-column": request.timestamp_column,
        "algorithm": request.algorithm,
        "k": str(request.k),
    }
    if request.hyperparameters_json is not None:
        parameters["hyperparameters-json"] = request.hyperparameters_json
    if request.rating_column is not None:
        parameters["rating-column"] = request.rating_column
    if request.item_features_uri is not None:
        parameters["item-features-uri"] = request.item_features_uri
    if request.item_id_column_features is not None:
        parameters["item-id-column-features"] = request.item_id_column_features
    if request.item_text_column is not None:
        parameters["item-text-column"] = request.item_text_column
    result = argo_adapter.trigger_workflow(REC_TRAIN_REGISTER_TEMPLATE, parameters)
    metadata = cast(dict[str, object], result["metadata"])
    return TriggerRecTrainingResponse(workflow_name=str(metadata["name"]))


@router.post("/rec-datasets/validate", response_model=list[CheckResultResponse])
def validate_rec_dataset(
    request: ValidateRecDatasetRequest, user: dict = Depends(get_current_user)
) -> list[CheckResultResponse]:
    csv_path = Path(request.interactions_uri.removeprefix("file://"))
    interactions = pd.read_csv(csv_path)
    results = run_rec_checks(interactions, request.user_id_column, request.item_id_column)
    return [CheckResultResponse.from_check_result(r) for r in results]
