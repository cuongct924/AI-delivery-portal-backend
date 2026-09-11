"""Model Registry / Training / Deploy-prep API — the HTTP surface Golden Path
#1 (Train -> Track -> Register) and #2 (Register -> Deploy) drive.

`POST /models/register` is the one route with no `Depends(get_current_user)`
— it's called from inside an Argo workflow pod, not from Backstage.
"""

from pathlib import Path
from typing import Final, cast

import pandas as pd
from auth.keycloak import get_current_user
from data_quality.checks import CheckResult
from data_quality.registry import run_checks
from evaluations.gate import MetricsGateResult, evaluate_metrics_gate
from fastapi import APIRouter, Depends
from jinja2 import Environment, FileSystemLoader
from kubernetes.client.exceptions import ApiException
from pydantic import BaseModel

from adapters.deploy_strategies import (
    DirectStrategy,
    InstantStrategy,
    PRGatedStrategy,
    TrafficSplitStrategy,
)
from adapters.factory import (
    get_feature_store_adapter,
    get_kserve_adapter,
    get_model_registry_adapter,
    get_object_storage_adapter,
    get_workflow_adapter,
)
from adapters.interfaces import DatasetInfo, IDeployTrafficStrategy, IReleaseStrategy

router = APIRouter(tags=["models"])

# Module-level singletons — same convention as
# agents/mcp-servers/observability-server/server.py.
mlflow_adapter = get_model_registry_adapter()
argo_adapter = get_workflow_adapter()
feast_adapter = get_feature_store_adapter()
object_storage_adapter = get_object_storage_adapter()

# One WorkflowTemplate covers both train and fine-tune; mode is a parameter.
TRAIN_REGISTER_TEMPLATE: Final[str] = "train-register-golden-path"

_TEMPLATES_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "templates"
_JINJA_ENV: Final[Environment] = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))


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
    # BYOC — only used when algorithm="custom".
    code_repo_url: str | None = None
    entrypoint_path: str | None = None
    custom_config: str | None = None
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


class WorkflowStatusResponse(BaseModel):
    name: str
    phase: str | None
    message: str | None


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


class PolicyCheckRequest(BaseModel):
    model_name: str
    model_version: str


class PrepareDeployRequest(BaseModel):
    model_name: str
    model_version: str
    # "direct" | "canary" | "ab" | "blue-green".
    traffic_strategy: str = "direct"
    traffic_percent: int | None = None
    # "pr-gated" | "instant"
    release_strategy: str = "pr-gated"


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


@router.post("/trigger-training", response_model=TriggerTrainingResponse)
def trigger_training(
    request: TriggerTrainingRequest, user: dict = Depends(get_current_user)
) -> TriggerTrainingResponse:
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
    if request.code_repo_url is not None:
        parameters["code-repo-url"] = request.code_repo_url
    if request.entrypoint_path is not None:
        parameters["entrypoint-path"] = request.entrypoint_path
    if request.custom_config is not None:
        parameters["custom-config"] = request.custom_config
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
    result = argo_adapter.trigger_workflow(TRAIN_REGISTER_TEMPLATE, parameters)
    metadata = cast(dict[str, object], result["metadata"])
    return TriggerTrainingResponse(workflow_name=str(metadata["name"]))


@router.get("/trigger-training/{workflow_name}/status", response_model=WorkflowStatusResponse)
def get_training_status(
    workflow_name: str, user: dict = Depends(get_current_user)
) -> WorkflowStatusResponse:
    status = argo_adapter.get_workflow_status(workflow_name)
    return WorkflowStatusResponse(**status)


@router.get("/trigger-training/recent", response_model=list[WorkflowSummary])
def list_recent_training_runs(user: dict = Depends(get_current_user)) -> list[WorkflowSummary]:
    return [
        WorkflowSummary(name=w.get("name"), phase=w.get("phase"), started_at=w.get("startedAt"))
        for w in argo_adapter.list_workflows()
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


@router.post("/datasets/validate", response_model=list[CheckResultResponse])
def validate_dataset(
    request: ValidateDatasetRequest, user: dict = Depends(get_current_user)
) -> list[CheckResultResponse]:
    csv_path = Path(request.dataset_uri.strip().removeprefix("file://"))
    df = pd.read_csv(csv_path)
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


@router.get("/models/{name}/{version}/summary", response_model=ModelVersionSummaryResponse)
def get_model_version_summary(
    name: str, version: str, user: dict = Depends(get_current_user)
) -> ModelVersionSummaryResponse:
    details = mlflow_adapter.get_model_version_details(name, version)
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


@router.post("/policy-check")
def policy_check(
    request: PolicyCheckRequest, user: dict = Depends(get_current_user)
) -> MetricsGateResult:
    # Classical ML has ground-truth metrics — compare directly, no LLM-as-judge.
    details = mlflow_adapter.get_model_version_details(request.model_name, request.model_version)
    task_type = details["tags"].get("task_type")
    if task_type is None:
        raise ValueError(
            f"model version {request.model_name}:{request.model_version} has no task_type tag "
            "— it was registered before task-type tagging was added"
        )
    gate_result = evaluate_metrics_gate(task_type, details["metrics"])

    # MLflow tags are strings — stringify every value before persisting.
    mlflow_adapter.set_model_version_tag(
        request.model_name, request.model_version, "gate_passed", str(gate_result["passed"])
    )
    for metric_name, value in details["metrics"].items():
        mlflow_adapter.set_model_version_tag(
            request.model_name, request.model_version, f"gate_{metric_name}", str(value)
        )
    return gate_result


@router.post("/deploy-model/prepare", response_model=PrepareDeployResponse)
def prepare_deploy_manifest(
    request: PrepareDeployRequest, user: dict = Depends(get_current_user)
) -> PrepareDeployResponse:
    # Canonical MLflow Model Registry URI — resolvable by any MLflow-aware loader.
    storage_uri = f"models:/{request.model_name}/{request.model_version}"

    # Lazy: KServeAdapter.__init__ eagerly calls load_kube_config(), which
    # would crash startup wherever no kubeconfig exists (CI, before `kind`).
    needs_kserve = request.traffic_strategy != "direct" or request.release_strategy == "instant"
    kserve_adapter = get_kserve_adapter("mlops-team") if needs_kserve else None

    traffic_strategy: IDeployTrafficStrategy
    if request.traffic_strategy == "direct":
        traffic_strategy = DirectStrategy()
    else:
        # Needs a prior deploy to compare/rollback against — enforced here
        # since the Scaffolder form can't gate on live cluster state.
        assert kserve_adapter is not None
        try:
            kserve_adapter.get_inference_status(request.model_name)
        except ApiException as exc:
            if exc.status != 404:
                raise
            raise ValueError(
                f"{request.model_name} has no prior deploy — "
                "choose deployStrategy=direct for a model's first deploy"
            ) from exc
        if request.traffic_percent is None:
            raise ValueError("traffic_percent is required when traffic_strategy is not 'direct'")
        traffic_strategy = TrafficSplitStrategy(request.traffic_percent)

    traffic_fields = traffic_strategy.render()
    template = _JINJA_ENV.get_template("inference_service.yaml.j2")
    content = template.render(
        model_name=request.model_name,
        model_version=request.model_version,
        storage_uri=storage_uri,
        canary_traffic_percent=traffic_fields.get("canaryTrafficPercent"),
    )
    # Always dev — this Golden Path is mlops-team's, and orchestration-api
    # never writes anywhere but dev (staging/prod promotion is
    # DeploymentPipeline-only, see infra/openchoreo/deployment-pipeline.yaml).
    file_name = (
        f"infra/environments/dev/inference-services/mlops-team/"
        f"{request.model_name}/{request.model_version}.yaml"
    )

    release_strategy: IReleaseStrategy
    if request.release_strategy == "instant":
        assert kserve_adapter is not None
        release_strategy = InstantStrategy(kserve_adapter, traffic_fields)
    else:
        release_strategy = PRGatedStrategy()
    release_result = release_strategy.release(request.model_name, request.model_version, content)

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
