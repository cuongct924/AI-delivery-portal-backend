"""Model Registry / Training / Deploy-prep API — the HTTP surface Golden
Path (Train -> Track -> Register) and (Register -> Deploy) drive.

`POST /models/register` is the one route with no `Depends(get_current_user)`
— it's called from inside an Argo workflow pod, not from Backstage.
"""

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Final, cast

import pandas as pd
from auth.thunder import get_current_user
from costs.events import record_cost_event
from costs.pricing import CPU_HOUR_USD, train_estimate_hours
from data_quality.checks import CheckResult
from data_quality.registry import run_checks
from evaluations.evaluate_gate import MetricsGateResult, evaluate_metrics_gate
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from idempotency import get_or_compute
from jinja2 import Environment, FileSystemLoader
from observability.dora_metrics import (
    DEPLOYMENT_EVENTS,
    GATE_EVALUATIONS,
    INCIDENT_RECOVERY,
    record_workflow_completion,
)
from pydantic import BaseModel

from adapters.ai_platform.interfaces import DatasetInfo
from adapters.delivery.deploy_strategies import (
    BlueGreenStrategy,
    DirectStrategy,
    InstantStrategy,
    PRGatedStrategy,
)
from adapters.delivery.interfaces import IDeployTrafficStrategy, IReleaseStrategy
from adapters.factory import (
    get_deployment_event_store,
    get_eval_result_adapter,
    get_feature_store_adapter,
    get_inference_adapter,
    get_inference_backend_mode,
    get_model_registry_adapter,
    get_object_storage_adapter,
    get_prediction_log_adapter,
    get_promotion_adapter,
    get_workflow_adapter,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Module-level singletons — same convention as
# agents/mcp-servers/ai-observability-server/server.py.
mlflow_adapter = get_model_registry_adapter()
workflow_adapter = get_workflow_adapter()
prediction_log_adapter = get_prediction_log_adapter()
promotion_adapter = get_promotion_adapter()
feast_adapter = get_feature_store_adapter()
object_storage_adapter = get_object_storage_adapter()
eval_result_adapter = get_eval_result_adapter()
deployment_event_store = get_deployment_event_store()

# One WorkflowTemplate covers both train and fine-tune; mode is a parameter.
TRAIN_REGISTER_TEMPLATE: Final[str] = "train-register-golden-path"

# RQ1 dora_metrics label — `get_training_status` is polled repeatedly by the
# frontend until the workflow reaches a terminal phase; this set stops a
# completion from being recorded more than once per workflow.
_RECORDED_TERMINAL_WORKFLOWS: set[str] = set()

# trigger_training's request carries model_name, but the polled
# get_training_status(workflow_name) call that later records the Delivery
# Insights deployment event does not — bridge the two by workflow_name.
_TRAINING_MODEL_NAMES: dict[str, str] = {}

_TEMPLATES_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "templates"
_JINJA_ENV: Final[Environment] = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))

# Mirrors OpenChoreoInferenceAdapter's __init__ defaults but duplicated as
# plain constants so rendering a PR-gated manifest never needs a kubeconfig.
_OPENCHOREO_PROJECT: Final[str] = "telco-fraud-detection"
_OPENCHOREO_COMPONENT: Final[str] = "serving"


class TriggerTrainingRequest(BaseModel):
    model_name: str
    dataset_uri: str
    task_type: str
    # sklearn by default — "algorithm" only applies to that architecture;
    # mlp/lstm use the DL hyperparameter fields below instead.
    architecture: str = "sklearn"
    algorithm: str | None = None
    target_column: str | None = None
    id_columns: list[str] | None = None
    time_column: str | None = None
    base_model_uri: str | None = None
    # DL hyperparameters — unused for architecture="sklearn".
    hidden_layers: list[int] | None = None
    dropout: float | None = None
    sequence_length: int | None = None
    num_layers: int | None = None
    hidden_size: int | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    # Dev-facing optimizer choice ("adam"/"sgd", optimizers.py) — only used
    # for architecture="mlp"/"lstm"/"nlp"/"cv", defaults to "adam" when unset.
    optimizer: str | None = None
    # HPO — only used when architecture is "mlp"/"lstm" and search_strategy
    # is not "fixed" (the default).
    search_strategy: str | None = None
    num_trials: int | None = None
    search_space_json: str | None = None
    objective_metric: str | None = None
    objective_direction: str | None = None
    # NLP — only used when architecture="nlp".
    text_column: str | None = None
    base_model_name: str | None = None
    # CV — no new fields, DATASET_URI/LEARNING_RATE/EPOCHS/BATCH_SIZE are
    # all reused as-is.


class TriggerTrainingResponse(BaseModel):
    workflow_name: str


class DatasetColumnsResponse(BaseModel):
    columns: list[str]


class FeatureListResponse(BaseModel):
    features: list[str]


class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, object]]


class ListDatasetsResponse(BaseModel):
    datasets: list[DatasetInfo]


class ValidateDatasetRequest(BaseModel):
    dataset_uri: str
    task_type: str
    target_column: str | None = None
    time_column: str | None = None


class EnrichDatasetFeaturesRequest(BaseModel):
    dataset_uri: str
    entity_id_column: str
    # Feast "<feature_view>:<feature>" references, e.g. "transaction_features:amount".
    feature_names: list[str]


class EnrichDatasetFeaturesResponse(BaseModel):
    dataset_uri: str


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


class WorkflowStepTimingResponse(BaseModel):
    name: str
    phase: str | None
    started_at: str | None
    finished_at: str | None


class WorkflowStatusResponse(BaseModel):
    name: str
    phase: str | None
    message: str | None
    started_at: str | None = None
    finished_at: str | None = None
    steps: list[WorkflowStepTimingResponse] = []


class WorkflowSummary(BaseModel):
    name: str | None
    phase: str | None
    started_at: str | None


class RegisterModelRequest(BaseModel):
    name: str
    artifact_uri: str
    task_type: str
    dataset_version: str | None = None


class RegisterModelResponse(BaseModel):
    name: str
    version: str


class ModelSummary(BaseModel):
    name: str
    version: str
    metrics: dict[str, float]
    tags: dict[str, str]


class ModelVersionSummaryResponse(BaseModel):
    name: str
    version: str
    task_type: str | None
    metrics: dict[str, float]
    tags: dict[str, str]


class LatestVersionResponse(BaseModel):
    name: str
    version: str


class ModelVersionsResponse(BaseModel):
    versions: list[str]


class PolicyCheckRequest(BaseModel):
    model_name: str
    model_version: str


class PrepareDeployRequest(BaseModel):
    model_name: str
    # For action="rollback", the version to roll back *to* — same field,
    # so the rest of this request/the release strategies below don't need
    # to know "rollback" exists as a concept at all.
    model_version: str
    # "direct" | "blue-green".
    traffic_strategy: str = "direct"
    # "pr-gated" | "instant"
    release_strategy: str = "pr-gated"
    # "deploy" | "rollback". Rollback is a production emergency — Dev picks
    # only the version; the instant 100% cutover is the platform's call, so
    # the strategy fields are ignored when action="rollback".
    action: str = "deploy"
    # Audit tag for the version; nothing here proxies predict traffic —
    # POST /models/{name}/predictions/log is the real logging entry point.
    enable_prediction_logging: bool = True


class PrepareDeployResponse(BaseModel):
    file_name: str
    content: str
    deployed: bool = False


class RecordDeployRequest(BaseModel):
    model_name: str
    model_version: str
    pr_url: str | None = None


class RecordDeployResponse(BaseModel):
    model_name: str
    model_version: str
    pr_url: str | None = None


class DeployStatusResponse(BaseModel):
    deployed: bool
    ready: bool = False
    live_version: str | None = None
    traffic_percent: int | None = None
    pr_url: str | None = None


class LogPredictionRequest(BaseModel):
    model_version: str
    input: dict[str, object]
    output: dict[str, object] | None = None


class PredictionLogEntryResponse(BaseModel):
    id: int
    model_version: str
    logged_at: str
    input: dict[str, object]
    output: dict[str, object] | None


class ListPredictionsResponse(BaseModel):
    predictions: list[PredictionLogEntryResponse]


class PromotionStatusResponse(BaseModel):
    project: str
    component: str
    environments: dict[str, str | None]
    prod_pending_approval: bool


class PromoteRequest(BaseModel):
    target_environment: str


class RollbackPromotionRequest(BaseModel):
    environment: str


class PreparePromotionResponse(BaseModel):
    """Rendered ProjectReleaseBinding manifest — nothing has been written
    to the cluster yet. A Scaffolder step publishes `content` as a PR;
    merging it is the review, and POST .../promote/confirm (below) is what
    actually applies it."""

    file_name: str
    content: str
    environment: str
    project_release: str


class ConfirmPromotionRequest(BaseModel):
    environment: str
    project_release: str
    # "deploy" for a promotion, "rollback" for a rollback — only affects
    # the DORA deployment-event label, both call the same adapter method.
    event_type: str = "deploy"


@router.post("/trigger-training", response_model=TriggerTrainingResponse)
def trigger_training(
    request: TriggerTrainingRequest,
    user: dict = Depends(get_current_user),
    idempotency_key: str | None = Header(default=None),
) -> TriggerTrainingResponse:
    """`idempotency_key` (the `Idempotency-Key` header) guards against a
    retried request (Scaffolder step retry, double-click, network timeout)
    starting a *second* real training run — trigger_workflow() names each
    WorkflowRun from a timestamp, so a naive retry would never collide on
    its own. Sending the Scaffolder task's own `ctx.taskId` as the key
    makes retries of the same task idempotent (see mlopsActions.ts)."""
    parameters = {
        "model-name": request.model_name,
        "dataset-uri": request.dataset_uri.strip(),
        "task-type": request.task_type,
        "architecture": request.architecture,
        "mode": "finetune" if request.base_model_uri is not None else "train",
    }
    if request.algorithm is not None:
        parameters["algorithm"] = request.algorithm
    if request.target_column is not None:
        parameters["target-column"] = request.target_column
    if request.id_columns:
        parameters["id-columns"] = ",".join(request.id_columns)
    if request.time_column is not None:
        parameters["time-column"] = request.time_column
    if request.base_model_uri is not None:
        parameters["base-model-uri"] = request.base_model_uri
    if request.hidden_layers is not None:
        parameters["hidden-layers"] = ",".join(str(n) for n in request.hidden_layers)
    if request.dropout is not None:
        parameters["dropout"] = str(request.dropout)
    if request.sequence_length is not None:
        parameters["sequence-length"] = str(request.sequence_length)
    if request.num_layers is not None:
        parameters["num-layers"] = str(request.num_layers)
    if request.hidden_size is not None:
        parameters["hidden-size"] = str(request.hidden_size)
    if request.learning_rate is not None:
        parameters["learning-rate"] = str(request.learning_rate)
    if request.epochs is not None:
        parameters["epochs"] = str(request.epochs)
    if request.batch_size is not None:
        parameters["batch-size"] = str(request.batch_size)
    if request.optimizer is not None:
        parameters["optimizer"] = request.optimizer
    if request.search_strategy is not None:
        parameters["search-strategy"] = request.search_strategy
    if request.num_trials is not None:
        parameters["num-trials"] = str(request.num_trials)
    if request.search_space_json is not None:
        parameters["search-space-json"] = request.search_space_json
    if request.objective_metric is not None:
        parameters["objective-metric"] = request.objective_metric
    if request.objective_direction is not None:
        parameters["objective-direction"] = request.objective_direction
    if request.text_column is not None:
        parameters["text-column"] = request.text_column
    if request.base_model_name is not None:
        parameters["base-model-name"] = request.base_model_name

    def _do_trigger() -> TriggerTrainingResponse:
        result = workflow_adapter.trigger_workflow(TRAIN_REGISTER_TEMPLATE, parameters)
        metadata = cast(dict[str, object], result["metadata"])
        workflow_name = str(metadata["name"])
        _TRAINING_MODEL_NAMES[workflow_name] = request.model_name
        # Attribute the run's compute to the model's build stage. The real
        # duration is only known once the workflow finishes, so this prices a
        # nominal run length scaled by the form's own cost drivers (epochs,
        # HPO trials) — source=workflow-estimate keeps that honest.
        hours = train_estimate_hours(request.epochs, request.num_trials)
        record_cost_event(
            stage="build",
            artifact_kind="model",
            artifact_id=request.model_name,
            environment="development",
            cost_usd=hours * CPU_HOUR_USD,
            quantity=hours,
            unit="cpu-hour",
            unit_price=CPU_HOUR_USD,
            source="workflow-estimate",
            run_id=workflow_name,
        )
        return TriggerTrainingResponse(workflow_name=workflow_name)

    return get_or_compute(idempotency_key, _do_trigger)


@router.get("/trigger-training/{workflow_name}/status", response_model=WorkflowStatusResponse)
def get_training_status(
    workflow_name: str, user: dict = Depends(get_current_user)
) -> WorkflowStatusResponse:
    status = workflow_adapter.get_workflow_status(workflow_name)
    phase = status.get("phase")
    if phase in ("Succeeded", "Failed") and workflow_name not in _RECORDED_TERMINAL_WORKFLOWS:
        _RECORDED_TERMINAL_WORKFLOWS.add(workflow_name)
        record_workflow_completion(
            golden_path="train-track-register",
            phase=phase,
            started_at=status.get("started_at"),
            finished_at=status.get("finished_at"),
            steps=status.get("steps"),
        )
        started_at = status.get("started_at")
        finished_at = status.get("finished_at")
        if started_at and finished_at:
            deployment_event_store.record_event(
                name=workflow_name,
                change_type="model",
                project_name=_TRAINING_MODEL_NAMES.get(workflow_name, workflow_name),
                component_name="train-track-register",
                environment_name="development",
                outcome="success" if phase == "Succeeded" else "failed",
                started_at=started_at,
                finished_at=finished_at,
                steps=status.get("steps"),
            )
    return WorkflowStatusResponse(
        name=status["name"],
        phase=status.get("phase"),
        message=status.get("message"),
        started_at=status.get("started_at"),
        finished_at=status.get("finished_at"),
        steps=[WorkflowStepTimingResponse(**step) for step in status.get("steps", [])],
    )


@router.get("/trigger-training/recent", response_model=list[WorkflowSummary])
def list_recent_training_runs(user: dict = Depends(get_current_user)) -> list[WorkflowSummary]:
    return [
        WorkflowSummary(name=w.get("name"), phase=w.get("phase"), started_at=w.get("startedAt"))
        for w in workflow_adapter.list_workflows()
    ]


@router.post("/models/register", response_model=RegisterModelResponse)
def register_model(request: RegisterModelRequest) -> RegisterModelResponse:
    result = mlflow_adapter.register_model(
        request.name, request.artifact_uri, request.dataset_version
    )
    # Tagged separately so policy_check() can read it back at deploy time.
    mlflow_adapter.set_model_version_tag(
        result["name"], result["version"], "task_type", request.task_type
    )
    return RegisterModelResponse(**result)


@router.get("/datasets", response_model=ListDatasetsResponse)
def list_datasets(user: dict = Depends(get_current_user)) -> ListDatasetsResponse:
    """Lists datasets already pushed to the object store, for the
    Scaffolder UI's dataset picker (StepLayoutField's `datasetPicker`).
    """
    return ListDatasetsResponse(datasets=object_storage_adapter.list_datasets())


@router.get("/datasets/columns", response_model=DatasetColumnsResponse)
def get_dataset_columns(
    dataset_uri: str, user: dict = Depends(get_current_user)
) -> DatasetColumnsResponse:
    """Reads a dataset's CSV header so the Scaffolder UI can offer real column names.

    Backs StepLayoutField's column pickers (Target column, ID columns,
    Time column, Text column) instead of making the user guess/type column
    names blind. A dataset that isn't a CSV (architecture=cv's `.zip` of
    images) legitimately fails here — the frontend falls back to a plain
    text/array input in that case.
    """
    csv_path = Path(dataset_uri.strip().removeprefix("file://"))
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return DatasetColumnsResponse(columns=columns)


@router.get("/datasets/preview", response_model=DatasetPreviewResponse)
def preview_dataset(
    dataset_uri: str,
    limit: int = Query(default=20, ge=1, le=100),
    user: dict = Depends(get_current_user),
) -> DatasetPreviewResponse:
    """Reads a dataset's first `limit` rows so the Scaffolder UI can show what
    the data actually looks like before training — same fail-open contract as
    get_dataset_columns above: a non-CSV dataset (architecture=cv's `.zip` of
    images) legitimately fails here, and the frontend just hides the preview.

    Goes through `df.to_json`/`json.loads` rather than `df.to_dict` directly:
    pandas' own JSON encoder turns NaN (missing values are the norm in real
    datasets) into `null`, which `dict` leaves as a float `nan` — invalid
    JSON that `json.dumps` would happily emit anyway (non-compliant `NaN`
    tokens) and browsers' `JSON.parse` then rejects.
    """
    csv_path = Path(dataset_uri.strip().removeprefix("file://"))
    df = pd.read_csv(csv_path, nrows=limit)
    # to_json only returns None when writing to a path_or_buf, which we don't pass.
    rows = cast(list[dict[str, object]], json.loads(cast(str, df.to_json(orient="records"))))
    return DatasetPreviewResponse(columns=df.columns.tolist(), rows=rows)


@router.post("/datasets/validate", response_model=list[CheckResultResponse])
def validate_dataset(
    request: ValidateDatasetRequest, user: dict = Depends(get_current_user)
) -> list[CheckResultResponse]:
    csv_path = Path(request.dataset_uri.strip().removeprefix("file://"))
    df = pd.read_csv(csv_path)
    # A stale form value must surface as a clear 400, not a 500 KeyError.
    for column, label in (
        (request.target_column, "target_column"),
        (request.time_column, "time_column"),
    ):
        if column is not None and column not in df.columns:
            raise HTTPException(
                400,
                f"{label} {column!r} is not a column in this dataset — "
                f"available columns: {df.columns.tolist()}",
            )
    results = run_checks(df, request.task_type, request.target_column, request.time_column)
    return [CheckResultResponse.from_check_result(r) for r in results]


@router.post("/datasets/enrich-features", response_model=EnrichDatasetFeaturesResponse)
def enrich_dataset_features(
    request: EnrichDatasetFeaturesRequest, user: dict = Depends(get_current_user)
) -> EnrichDatasetFeaturesResponse:
    csv_path = Path(request.dataset_uri.strip().removeprefix("file://"))
    df = pd.read_csv(csv_path)
    entity_ids = df[request.entity_id_column].astype(str).tolist()

    features = feast_adapter.get_offline_features(entity_ids, request.feature_names)
    # Drop Feast's own "event_timestamp" — only entity_id + features are kept.
    feature_columns = [name.split(":", 1)[1] for name in request.feature_names]
    features_df = pd.DataFrame(features)[["entity_id", *feature_columns]]

    # Feast's values take precedence over any same-named column already in
    # the dataset — otherwise pandas silently suffixes both as _x/_y.
    df = df.drop(columns=[c for c in feature_columns if c in df.columns])
    df["_feast_entity_id"] = df[request.entity_id_column].astype(str)
    enriched = df.merge(features_df, left_on="_feast_entity_id", right_on="entity_id", how="left")
    enriched = enriched.drop(columns=["_feast_entity_id", "entity_id"])

    enriched_path = csv_path.with_stem(f"{csv_path.stem}-enriched")
    enriched.to_csv(enriched_path, index=False)
    return EnrichDatasetFeaturesResponse(dataset_uri=f"file://{enriched_path}")


@router.get("/features", response_model=FeatureListResponse)
def list_available_features(user: dict = Depends(get_current_user)) -> FeatureListResponse:
    """Lists every `<feature_view>:<feature>` this Feast store actually
    has, for the Scaffolder UI's Feature names picker (StepLayoutField's
    Feature Enrichment panel) — same fail-open contract as the dataset
    endpoints above: a Feast repo that isn't `feast apply`-ed yet
    shouldn't 500 the whole form, just leave the picker empty (frontend
    falls back to a plain text input).
    """
    try:
        return FeatureListResponse(features=feast_adapter.list_available_features())
    except Exception:
        logger.warning(
            "list_available_features failed — Feast repo not applied yet?", exc_info=True
        )
        return FeatureListResponse(features=[])


@router.get("/models/{name}/{version}/summary", response_model=ModelVersionSummaryResponse)
def get_model_version_summary(
    name: str, version: str, user: dict = Depends(get_current_user)
) -> ModelVersionSummaryResponse:
    # 404-not-500: the Scaffolder's live version picker needs a clean 404 on
    # a typo'd version, not an unhandled 500.
    try:
        details = mlflow_adapter.get_model_version_details(name, version)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return ModelVersionSummaryResponse(
        name=name,
        version=details["version"],
        task_type=details["tags"].get("task_type"),
        metrics=details["metrics"],
        tags=details["tags"],
    )


@router.get("/models", response_model=list[ModelSummary])
def list_models(user: dict = Depends(get_current_user)) -> list[ModelSummary]:
    summaries: list[ModelSummary] = []
    for model in mlflow_adapter.list_models():
        name = model["name"]
        try:
            version = mlflow_adapter.get_latest_version(name)
        except ValueError:
            # Registered with zero versions yet (e.g. mid-training) — skip
            # rather than failing the whole Dashboard listing.
            continue
        details = mlflow_adapter.get_model_version_details(name, version)
        summaries.append(
            ModelSummary(
                name=name,
                version=details["version"],
                metrics=details["metrics"],
                tags=details["tags"],
            )
        )
    return summaries


@router.get("/models/{name}/latest-version", response_model=LatestVersionResponse)
def get_latest_version(name: str, user: dict = Depends(get_current_user)) -> LatestVersionResponse:
    return LatestVersionResponse(name=name, version=mlflow_adapter.get_latest_version(name))


@router.get("/models/{name}/versions", response_model=ModelVersionsResponse)
def list_model_versions(name: str, user: dict = Depends(get_current_user)) -> ModelVersionsResponse:
    """Every version actually registered for `name` (newest first), for the
    Evaluate & Deploy Model template's version dropdown (ModelVersionPickerField)
    — picking from real versions instead of typing removes the invalid-version
    class entirely. Fails open to `[]` for an unknown name rather than 404 —
    the dropdown just stays empty (falls back to free text) until modelName
    resolves to something real.
    """
    return ModelVersionsResponse(versions=mlflow_adapter.list_model_versions(name))


def _compute_gate_result(model_name: str, model_version: str) -> MetricsGateResult:
    # Classical ML has ground-truth metrics — no LLM-as-judge. Both failure
    # modes below are routine caller input, so a clean 404/400 beats a 500.
    # Shared by policy_check (persists tags) and get_gate_preview (pure read).
    try:
        details = mlflow_adapter.get_model_version_details(model_name, model_version)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    task_type = details["tags"].get("task_type")
    if task_type is None:
        raise HTTPException(
            400,
            f"model version {model_name}:{model_version} has no task_type tag "
            "— it was registered before task-type tagging was added",
        )
    return evaluate_metrics_gate(task_type, details["metrics"])


@router.post("/policy-check")
def policy_check(
    request: PolicyCheckRequest, user: dict = Depends(get_current_user)
) -> MetricsGateResult:
    gate_result = _compute_gate_result(request.model_name, request.model_version)

    # MLflow tags are strings — stringify every value before persisting.
    mlflow_adapter.set_model_version_tag(
        request.model_name, request.model_version, "gate_passed", str(gate_result["passed"])
    )
    for metric_name, value in gate_result["metrics"].items():
        mlflow_adapter.set_model_version_tag(
            request.model_name, request.model_version, f"gate_{metric_name}", str(value)
        )

    # Emit DORA gate evaluation metric (MLOps track)
    GATE_EVALUATIONS.labels(
        track="mlops",
        subject_type="model",
        subject_id=f"{request.model_name}:{request.model_version}",
        passed=str(gate_result["passed"]).lower(),
    ).inc()

    return gate_result


@router.get("/models/{name}/{version}/gate-preview")
def get_gate_preview(
    name: str, version: str, user: dict = Depends(get_current_user)
) -> MetricsGateResult:
    """Read-only preview of what POST /policy-check would compute — same
    thresholds, no tag-writing side effect, safe to call repeatedly while
    a Dev is still filling in Evaluate & Deploy Model's form (see
    StepLayoutField's ModelVersionCheckPanel). The real Evaluate Gate step
    still re-runs this via POST /policy-check at submit time and persists
    gate_passed/gate_<metric> tags then — this is advisory only, same
    "advisory, the real gate is elsewhere" contract as
    ModelVersionCheckPanel/VersionComparisonPanel's own live panels."""
    return _compute_gate_result(name, version)


@router.post("/deploy-model/prepare", response_model=PrepareDeployResponse)
def prepare_deploy_manifest(
    request: PrepareDeployRequest, user: dict = Depends(get_current_user)
) -> PrepareDeployResponse:
    # Canonical MLflow Model Registry URI — resolvable by any MLflow-aware loader.
    storage_uri = f"models:/{request.model_name}/{request.model_version}"

    # Rollback overrides the strategy fields — Dev chose only the version, the
    # instant blue-green cutover is the platform's call. Locals, not request
    # mutation, so PrepareDeployRequest.action's contract stays visible here.
    is_rollback = request.action == "rollback"
    traffic_strategy_value = "blue-green" if is_rollback else request.traffic_strategy
    release_strategy_value = "instant" if is_rollback else request.release_strategy

    # Lazy: OpenChoreoInferenceAdapter.__init__ eagerly calls load_kube_config(),
    # which would crash startup wherever no kubeconfig exists (CI, before `kind`).
    needs_inference_adapter = (
        traffic_strategy_value != "direct" or release_strategy_value == "instant"
    )
    inference_adapter = get_inference_adapter("mlops-team") if needs_inference_adapter else None

    traffic_strategy: IDeployTrafficStrategy
    if traffic_strategy_value == "direct":
        traffic_strategy = DirectStrategy()
    else:
        # Needs a prior deploy to compare/rollback against — the form can't
        # gate on live cluster state; this also rejects empty rollbacks.
        assert inference_adapter is not None
        if not inference_adapter.get_deploy_status(request.model_name)["deployed"]:
            message = (
                f"{request.model_name} has no prior deploy — nothing to roll back to"
                if is_rollback
                else f"{request.model_name} has no prior deploy — "
                "choose deployStrategy=direct for a model's first deploy"
            )
            raise ValueError(message)
        traffic_strategy = BlueGreenStrategy()

    traffic_fields = traffic_strategy.render()
    canary_percent = traffic_fields.get("canaryTrafficPercent")
    backend_mode = get_inference_backend_mode()

    if backend_mode == "openchoreo":
        # The one live Workload file is the git-tracked source of truth — a
        # PR-gated deploy updates THIS file's storageUri, so merging the PR
        # is the approval and `git log` is the deploy history.
        template = _JINJA_ENV.get_template("workload.yaml.j2")
        content = template.render(
            project=_OPENCHOREO_PROJECT,
            component=_OPENCHOREO_COMPONENT,
            workload_name=f"{_OPENCHOREO_COMPONENT}-workload",
            namespace="default",
            storage_uri=storage_uri,
        )
        file_name = (
            f"infra/openchoreo/namespaces/default/projects/{_OPENCHOREO_PROJECT}/"
            f"components/{_OPENCHOREO_COMPONENT}/workload-{_OPENCHOREO_COMPONENT}.yaml"
        )
    else:
        template = _JINJA_ENV.get_template("inference_service.yaml.j2")
        content = template.render(
            model_name=request.model_name,
            model_version=request.model_version,
            storage_uri=storage_uri,
            canary_traffic_percent=canary_percent,
        )
        # This Golden Path is mlops-team's; orchestration-api never writes
        # beyond dev (staging/prod promotion is DeploymentPipeline-only).
        file_name = (
            f"infra/environments/dev/inference-services/mlops-team/"
            f"{request.model_name}/{request.model_version}.yaml"
        )

    release_strategy: IReleaseStrategy
    if release_strategy_value == "instant":
        assert inference_adapter is not None
        release_strategy = InstantStrategy(inference_adapter, traffic_fields, mlflow_adapter)
    else:
        release_strategy = PRGatedStrategy()
    release_result = release_strategy.release(request.model_name, request.model_version, content)

    # Tag regardless of deployed vs PR-gated — a PR-gated deploy's version
    # is still "the one this Dev intended to have logging on" even before
    # the PR merges and someone applies the manifest by hand.
    mlflow_adapter.set_model_version_tag(
        request.model_name,
        request.model_version,
        "prediction_logging_enabled",
        str(request.enable_prediction_logging),
    )

    # Only an Instant release has actually happened yet — a PR-gated one is
    # still pending a human merge (and, beyond that, an out-of-process
    # watcher applying it), with no callback into this process either way,
    # so recording a delivery event for it here would claim a change that
    # may never land. Instant is also the only branch that reaches here
    # synchronously, same reasoning as promote_model's own recording below.
    if release_result["deployed"]:
        DEPLOYMENT_EVENTS.labels(
            track="mlops",
            subject_type="model",
            subject_id=request.model_name,
            event_type="rollback" if is_rollback else "deploy",
        ).inc()
        now = datetime.now().isoformat()
        deployment_event_store.record_event(
            name=f"{request.action}-{request.model_name}-{now}",
            change_type="model",
            project_name=request.model_name,
            component_name=_OPENCHOREO_COMPONENT,
            environment_name="development",
            outcome="success",
            started_at=now,
            finished_at=now,
        )

    return PrepareDeployResponse(
        file_name=file_name, content=content, deployed=release_result["deployed"]
    )


@router.post("/deploy-model/record", response_model=RecordDeployResponse)
def record_deploy(
    request: RecordDeployRequest, user: dict = Depends(get_current_user)
) -> RecordDeployResponse:
    # No PR for an Instant release — nothing to tag.
    if request.pr_url:
        mlflow_adapter.set_model_version_tag(
            request.model_name, request.model_version, "deploy_pr_url", request.pr_url
        )
    return RecordDeployResponse(
        model_name=request.model_name,
        model_version=request.model_version,
        pr_url=request.pr_url,
    )


@router.get("/models/{name}/deploy-status", response_model=DeployStatusResponse)
def get_deploy_status(name: str, user: dict = Depends(get_current_user)) -> DeployStatusResponse:
    """What's actually live right now, for the Model Registry's "Deploy
    status" — so a Dev opening Evaluate & Deploy Model sees the current
    state (version, traffic split) before picking a strategy, instead of
    guessing and getting rejected by prepare_deploy_manifest's own
    prior-deploy check.

    Lazy adapter construction, same reason as prepare_deploy_manifest:
    OpenChoreoInferenceAdapter.__init__ eagerly loads a kubeconfig, which
    would crash every other route in an environment with none (CI, before
    `kind`).
    """
    inference_adapter = get_inference_adapter("mlops-team")
    status = inference_adapter.get_deploy_status(name)
    if not status["deployed"]:
        return DeployStatusResponse(deployed=False)

    pr_url = None
    live_version = status["live_version"]
    if live_version is not None:
        # ValueError: the version survived on the deploy status but that
        # model version was since deleted from the registry.
        with contextlib.suppress(ValueError):
            pr_url = mlflow_adapter.get_model_version_details(name, live_version)["tags"].get(
                "deploy_pr_url"
            )

    return DeployStatusResponse(
        deployed=True,
        ready=status["ready"],
        live_version=live_version,
        traffic_percent=status["traffic_percent"],
        pr_url=pr_url,
    )


@router.post("/models/{name}/predictions/log")
def log_prediction(
    name: str, request: LogPredictionRequest, user: dict = Depends(get_current_user)
) -> None:
    """The actual data-collection entry point for Golden Path #4 (data
    drift monitoring), started now rather than waiting for that Golden
    Path to exist first. Not automatic: nothing in this codebase proxies
    real predict traffic (adapters/delivery/gpu_inference_adapter.py's own predict()
    explicitly tells callers to hit the InferenceService directly,
    IPredictionLogAdapter's docstring has the full reasoning) — whoever
    calls the deployed model directly calls this too, alongside it, to
    have that call logged. Always logs what it's given; the
    "prediction_logging_enabled" tag prepare_deploy_manifest sets is an
    audit record of the Dev's original choice, not a gate checked here.
    """
    prediction_log_adapter.log_prediction(
        name, request.model_version, request.input, request.output
    )


@router.get("/models/{name}/predictions", response_model=ListPredictionsResponse)
def list_predictions(
    name: str, limit: int = 50, user: dict = Depends(get_current_user)
) -> ListPredictionsResponse:
    entries = prediction_log_adapter.list_predictions(name, limit)
    return ListPredictionsResponse(
        predictions=[
            PredictionLogEntryResponse(
                id=entry["id"],
                model_version=entry["model_version"],
                logged_at=entry["logged_at"],
                input=entry["input"],
                output=entry["output"],
            )
            for entry in entries
        ]
    )


@router.get("/models/{name}/promotion-status", response_model=PromotionStatusResponse)
def get_promotion_status(
    name: str, user: dict = Depends(get_current_user)
) -> PromotionStatusResponse:
    """promotion_adapter is scoped to the one real Project/Component this
    repo has, same as OpenChoreoInferenceAdapter's own single-Component
    scoping (see that adapter's docstring) — `name` is checked against it
    rather than silently ignored, because the frontend's modelName picker
    lists every registered model, not just this one, and operating on the
    wrong project without saying so would be worse than a clear error."""
    status = promotion_adapter.get_promotion_status()
    if name != status["project"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' has no promotion pipeline — only '{status['project']}' does today",
        )
    return PromotionStatusResponse(**status)


def _render_promotion_manifest(
    project: str, environment: str, project_release: str
) -> PreparePromotionResponse:
    template = _JINJA_ENV.get_template("projectreleasebinding.yaml.j2")
    content = template.render(
        project=project, environment=environment, project_release=project_release
    )
    file_name = (
        f"infra/openchoreo/namespaces/default/projects/{project}/"
        f"projectreleasebinding-{environment}.yaml"
    )
    return PreparePromotionResponse(
        file_name=file_name,
        content=content,
        environment=environment,
        project_release=project_release,
    )


@router.post("/models/{name}/promote", response_model=PreparePromotionResponse)
def promote_model(
    name: str, request: PromoteRequest, user: dict = Depends(get_current_user)
) -> PreparePromotionResponse:
    """Renders the ProjectReleaseBinding manifest that would promote `name`
    to `request.target_environment` — writes nothing yet. A Scaffolder step
    publishes the result as a PR (same as dev's PRGatedStrategy); merging it
    is the review. Call POST .../promote/confirm once it's merged to
    actually apply it — see adapters/delivery/openchoreo_promotion_adapter.py's
    module docstring for why staging/prod promotion is PR-gated.
    Same `name` scope-check as get_promotion_status — see its docstring."""
    current = promotion_adapter.get_promotion_status()
    if name != current["project"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' has no promotion pipeline — only '{current['project']}' does today",
        )
    try:
        release = promotion_adapter.resolve_promotion_release(request.target_environment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _render_promotion_manifest(current["project"], request.target_environment, release)


@router.post("/models/{name}/promote-rollback", response_model=PreparePromotionResponse)
def rollback_promotion(
    name: str, request: RollbackPromotionRequest, user: dict = Depends(get_current_user)
) -> PreparePromotionResponse:
    """staging/prod counterpart to the dev-side action=rollback — renders
    the manifest that would undo the last promote()/confirm-promotion for
    one environment. Same PR-then-confirm flow and `name` scope-check as
    promote_model — see its docstring and
    adapters/delivery/openchoreo_promotion_adapter.py's."""
    current = promotion_adapter.get_promotion_status()
    if name != current["project"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' has no promotion pipeline — only '{current['project']}' does today",
        )
    try:
        release = promotion_adapter.resolve_rollback_release(request.environment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _render_promotion_manifest(current["project"], request.environment, release)


@router.post("/models/{name}/promote/confirm", response_model=PromotionStatusResponse)
def confirm_promotion(
    name: str,
    request: ConfirmPromotionRequest,
    user: dict = Depends(get_current_user),
    idempotency_key: str | None = Header(default=None),
) -> PromotionStatusResponse:
    """Applies a ProjectReleaseBinding rendered by promote_model()/
    rollback_promotion() — the actual OpenChoreo/cluster write, and the
    point where delivery-metrics events are recorded, since nothing was
    actually deployed at render time. Call this once a human has reviewed
    the manifest (e.g. merged the PR it was published as); the manual
    review of that PR *is* the approval gate — see
    adapters/delivery/openchoreo_promotion_adapter.py's module docstring."""
    current = promotion_adapter.get_promotion_status()
    if name != current["project"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' has no promotion pipeline — only '{current['project']}' does today",
        )

    def _do_confirm() -> PromotionStatusResponse:
        try:
            status = promotion_adapter.confirm_promotion(
                request.environment, request.project_release
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Emit DORA deployment event (MLOps track)
        DEPLOYMENT_EVENTS.labels(
            track="mlops",
            subject_type="model",
            subject_id=name,
            event_type=request.event_type,
        ).inc()
        _record_promotion_deployment_event(name, current["component"], request.environment)

        # MTTR: find last drift detection for this model and calculate recovery time
        last_drift = _get_last_drift_detected_at(name)
        if last_drift is not None:
            recovery_seconds = (datetime.now() - last_drift).total_seconds()
            INCIDENT_RECOVERY.labels(
                track="mlops",
                subject_type="model",
                subject_id=name,
            ).observe(recovery_seconds)

        return PromotionStatusResponse(**status)

    return get_or_compute(idempotency_key, _do_confirm)


def _record_promotion_deployment_event(
    project_name: str, component_name: str, environment_name: str
) -> None:
    """confirm_promotion() is a synchronous OpenChoreo ReleaseBinding swap,
    not a queued workflow — started/finished are the same instant, and
    there's no workflow-run name, so one is synthesized from the
    timestamp."""
    now = datetime.now().isoformat()
    deployment_event_store.record_event(
        name=f"promote-{project_name}-{now}",
        change_type="model",
        project_name=project_name,
        component_name=component_name,
        environment_name=environment_name,
        outcome="success",
        started_at=now,
        finished_at=now,
    )


def _get_last_drift_detected_at(model_name: str) -> datetime | None:
    """Returns the start_time of the most recent monitoring run with drift_detected=True."""
    filter_string = f"tags.monitoring_model_name = '{model_name}' and tags.drift_detected = 'True'"
    runs = mlflow_adapter.search_runs(
        filter_string=filter_string, order_by=["start_time DESC"], max_results=1
    )
    if runs.empty:
        return None
    start_time = runs.iloc[0]["start_time"]
    if isinstance(start_time, str):
        return datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    return start_time
