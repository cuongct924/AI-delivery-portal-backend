"""Golden Path #1 training entrypoint — reads job config from env vars (set
by the `train-step` container in infra/argo-workflows/train-register-template.yaml),
trains (or fine-tunes) the selected algorithm, logs everything to MLflow,
and hands the resulting artifact URI + dataset digest to `register-step` via
/tmp files (Argo reads them back through `outputs.parameters`).
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd

# Must import before xgboost/lightgbm — reverse order segfaults on macOS
# (OpenMP conflict), harmless on Linux.
import torch  # noqa: F401
from algorithm_registry import AlgorithmSpec, get_algorithm_spec
from byoc_runner import run_custom_training
from hpo_runner import build_search_spaces, run_hpo
from hpo_strategies import build_search_strategy
from metrics import compute_metrics

# Submodule imports — mlflow's top-level stub doesn't declare these as exported attributes.
from mlflow import data as mlflow_data
from mlflow import pyfunc as mlflow_pyfunc
from mlflow import pytorch as mlflow_pytorch
from mlflow import sklearn as mlflow_sklearn
from mlflow import transformers as mlflow_transformers
from pyfunc_wrapper import GenericPyfuncWrapper
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from train_cv import train_and_evaluate as train_cv_and_evaluate
from train_dl import train_and_evaluate as train_dl_and_evaluate
from train_nlp import train_and_evaluate as train_nlp_and_evaluate

# Below this row count, a single random holdout is too noisy to trust —
# k-fold cross-validation averages over multiple splits instead.
_SMALL_DATASET_THRESHOLD = 50
_HOLDOUT_TEST_SIZE = 0.3
_KFOLD_SPLITS = 5
_TIME_SERIES_SPLITS = 5


def _read_dataset_digest(csv_path: Path) -> str:
    """Reads the DVC md5 hash — the dataset-lineage convention documented
    in data/README.md."""
    dvc_path = csv_path.with_name(csv_path.name + ".dvc")
    dvc_text = dvc_path.read_text()
    md5_match = re.search(r"md5:\s*(\S+)", dvc_text)
    if md5_match is None:
        raise RuntimeError(f"no md5 hash found in {dvc_path}")
    return md5_match.group(1)


def _encode_categoricals(features: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encodes every object-dtype column so any registry algorithm
    (none of which parse strings) gets purely numeric input. NaN is kept as
    NaN (not the -1 sentinel `.cat.codes` would otherwise assign) so missing
    value handling downstream applies uniformly to encoded and
    already-numeric columns alike."""
    encoded = features.copy()
    for column in encoded.select_dtypes(include="object").columns:
        codes = encoded[column].astype("category").cat.codes.astype("float64")
        encoded[column] = codes.replace(-1.0, np.nan)
    return encoded


def _handle_missing_values(
    train_features: pd.DataFrame, test_features: pd.DataFrame, spec: AlgorithmSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputes with the train split's median unless the algorithm handles
    NaN natively (XGBoost/LightGBM/CatBoost — where missingness itself can
    be predictive, so imputing it away would destroy signal)."""
    if spec.handles_missing_natively:
        return train_features, test_features
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )
    test_imputed = pd.DataFrame(
        imputer.transform(test_features), columns=test_features.columns, index=test_features.index
    )
    return train_imputed, test_imputed


def _scale_features(
    train_features: pd.DataFrame, test_features: pd.DataFrame, spec: AlgorithmSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard-scales distance-/gradient-based estimators; leaves
    tree-based ones alone (they split on raw values, scaling is a no-op at
    best)."""
    if not spec.requires_scaling:
        return train_features, test_features
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_features), columns=test_features.columns, index=test_features.index
    )
    return train_scaled, test_scaled


def _split(
    df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    task_type: str,
    time_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Picks the validation strategy automatically — Dev never chooses this.

    A `time_column` always wins and never shuffles — random holdout/k-fold
    would let future rows leak into training and inflate metrics
    regardless of dataset size. Otherwise: k-fold (via cross-validation
    indices) for small datasets, plain holdout otherwise.
    """
    if time_column is not None:
        order = df[time_column].argsort()
        features = features.iloc[order]
        labels = labels.iloc[order]
        splitter = TimeSeriesSplit(n_splits=min(_TIME_SERIES_SPLITS, len(df) - 1))
        train_idx, test_idx = list(splitter.split(features))[-1]
        return (
            features.iloc[train_idx],
            features.iloc[test_idx],
            labels.iloc[train_idx],
            labels.iloc[test_idx],
        )
    if len(df) < _SMALL_DATASET_THRESHOLD:
        splitter = TimeSeriesSplit(n_splits=min(_KFOLD_SPLITS, len(df) - 1))
        # Reused as a "last N% held out" splitter here, not for time ordering.
        train_idx, test_idx = list(splitter.split(features))[-1]
        return (
            features.iloc[train_idx],
            features.iloc[test_idx],
            labels.iloc[train_idx],
            labels.iloc[test_idx],
        )
    stratify = labels if task_type == "classification" else None
    # train_test_split's stub returns a generic `list` (arg count is
    # variadic) — 2 arrays in always means this exact 4-tuple shape out.
    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            features, labels, test_size=_HOLDOUT_TEST_SIZE, random_state=42, stratify=stratify
        ),
    )


def _fit(
    spec: AlgorithmSpec,
    mode: str,
    base_model_uri: str | None,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
) -> Any:  # see AlgorithmSpec.estimator_class — no shared typed base class across libraries
    if mode == "train":
        model = spec.estimator_class()
        model.fit(train_features, train_labels)
        return model
    if mode == "finetune":
        if base_model_uri is None:
            raise RuntimeError("MODE=finetune requires BASE_MODEL_URI")
        # mlflow.sklearn's stub types load_model()'s return as None — wrong,
        # it returns the loaded estimator.
        model = cast(Any, mlflow_sklearn.load_model(base_model_uri))
        if not hasattr(model, "warm_start"):
            raise RuntimeError(
                f"{type(model).__name__} does not support warm_start — cannot fine-tune. "
                "Use MODE=train to train a new model instead."
            )
        model.set_params(warm_start=True)
        model.fit(train_features, train_labels)
        return model
    raise RuntimeError(f"unknown MODE {mode!r} — must be 'train' or 'finetune'")


def _read_dl_hyperparameters() -> dict[str, object]:
    """Reads the DL hyperparameter env vars set by train-register-template.yaml.
    Only LEARNING_RATE/EPOCHS/BATCH_SIZE are common to both architectures —
    the rest are architecture-specific and simply absent from the dict when
    unset, letting train_dl.py fail loudly via KeyError if a required one is
    genuinely missing instead of silently defaulting."""
    hyperparameters: dict[str, object] = {
        "learning_rate": float(os.environ["LEARNING_RATE"]),
        "epochs": int(os.environ["EPOCHS"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
    }
    if hidden_layers := os.environ.get("HIDDEN_LAYERS"):
        hyperparameters["hidden_layers"] = [int(n) for n in hidden_layers.split(",") if n]
    if dropout := os.environ.get("DROPOUT"):
        hyperparameters["dropout"] = float(dropout)
    if sequence_length := os.environ.get("SEQUENCE_LENGTH"):
        hyperparameters["sequence_length"] = int(sequence_length)
    if num_layers := os.environ.get("NUM_LAYERS"):
        hyperparameters["num_layers"] = int(num_layers)
    if hidden_size := os.environ.get("HIDDEN_SIZE"):
        hyperparameters["hidden_size"] = int(hidden_size)
    # Absent means train_dl.py/train_cv.py's own "adam" default applies.
    if optimizer := os.environ.get("OPTIMIZER"):
        hyperparameters["optimizer"] = optimizer
    return hyperparameters


def _read_nlp_hyperparameters() -> dict[str, object]:
    """Reads the NLP hyperparameter env vars. LEARNING_RATE/EPOCHS/
    BATCH_SIZE are the same workflow parameters the DL path uses — reused
    as-is since the 2 architectures never run in the same job."""
    hyperparameters: dict[str, object] = {
        "base_model_name": os.environ["BASE_MODEL_NAME"],
        "learning_rate": float(os.environ["LEARNING_RATE"]),
        "epochs": int(os.environ["EPOCHS"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
    }
    if optimizer := os.environ.get("OPTIMIZER"):
        hyperparameters["optimizer"] = optimizer
    return hyperparameters


def main() -> None:
    dataset_uri = os.environ["DATASET_URI"]
    task_type = os.environ["TASK_TYPE"]
    target_column = os.environ.get("TARGET_COLUMN") or None
    id_columns = [c for c in os.environ.get("ID_COLUMNS", "").split(",") if c]
    architecture = os.environ.get("ARCHITECTURE") or "sklearn"
    algorithm = os.environ.get("ALGORITHM") or None
    mode = os.environ.get("MODE", "train")
    base_model_uri = os.environ.get("BASE_MODEL_URI") or None
    time_column = os.environ.get("TIME_COLUMN") or None
    # "custom" bypasses the algorithm registry and DL architecture registry
    # entirely, so it's checked ahead of both.
    is_custom = algorithm == "custom"
    code_repo_url = os.environ.get("CODE_REPO_URL") or None
    entrypoint_path = os.environ.get("ENTRYPOINT_PATH") or None
    # "fixed" (default) keeps the non-search code path completely
    # unchanged — no nested runs, no Optuna involved at all.
    search_strategy_name = os.environ.get("SEARCH_STRATEGY") or "fixed"
    is_search = search_strategy_name != "fixed"
    # text_column stays out of _encode_categoricals()'s reach (see the elif
    # branch below) so the tokenizer gets raw strings.
    is_nlp = architecture == "nlp"
    text_column = os.environ.get("TEXT_COLUMN") or None
    # DATASET_URI is a .zip of images for this architecture, not a CSV.
    is_cv = architecture == "cv"

    if is_nlp and text_column is None:
        raise RuntimeError("TEXT_COLUMN is required when ARCHITECTURE=nlp")
    if is_nlp and task_type != "classification":
        raise RuntimeError("ARCHITECTURE=nlp only supports TASK_TYPE=classification")
    if is_nlp and mode != "train":
        raise RuntimeError("ARCHITECTURE=nlp does not support MODE=finetune")
    if is_cv and task_type != "classification":
        raise RuntimeError("ARCHITECTURE=cv only supports TASK_TYPE=classification")
    if is_cv and mode != "train":
        raise RuntimeError("ARCHITECTURE=cv does not support MODE=finetune")

    if is_custom and (code_repo_url is None or entrypoint_path is None):
        raise RuntimeError("CODE_REPO_URL and ENTRYPOINT_PATH are required when ALGORITHM=custom")
    if is_custom and mode != "train":
        raise RuntimeError("BYOC (ALGORITHM=custom) does not support MODE=finetune")
    # anomaly-detection's target column is optional (evaluation only —
    # IsolationForest/LocalOutlierFactor never train on it, see the
    # anomaly-detection branch below), same as clustering having none at all.
    if not is_cv and task_type not in ("clustering", "anomaly-detection") and target_column is None:
        raise RuntimeError(f"TARGET_COLUMN is required for task_type {task_type!r}")
    if not is_custom and architecture == "sklearn" and algorithm is None:
        raise RuntimeError("ALGORITHM is required when ARCHITECTURE=sklearn")
    if (
        not is_custom
        and architecture != "sklearn"
        and task_type in ("clustering", "anomaly-detection")
    ):
        # dl_architecture_registry.py's DL_ARCHITECTURES only lists
        # classification/regression hyperparameters — no DL clustering or
        # anomaly-detection support.
        raise RuntimeError(
            f"architecture {architecture!r} does not support task_type={task_type!r}"
        )
    if is_search and (is_custom or is_nlp or is_cv or architecture == "sklearn"):
        # HPO is scoped to the DL hyperparameters — the only ones with an
        # existing single-value form field to search over.
        raise RuntimeError("SEARCH_STRATEGY != 'fixed' requires ARCHITECTURE=mlp or lstm")

    # Strip the "file://" scheme to get a real filesystem path.
    dataset_path = Path(dataset_uri.removeprefix("file://"))
    dataset_digest = _read_dataset_digest(dataset_path)

    if is_cv:
        # No DataFrame for CV — ImageFolder reads straight from the
        # extracted zip inside train_cv.py.
        df = None
        features = None
    else:
        df = pd.read_csv(dataset_path)
        drop_columns = list(id_columns)
        if target_column is not None:
            drop_columns.append(target_column)
        features = _encode_categoricals(df.drop(columns=drop_columns))

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
    )

    # Opened before fitting (not just before logging) so train_dl.py's
    # per-epoch mlflow.log_metric(..., step=epoch) calls land in this run.
    with mlflow.start_run() as run:
        mlflow.set_tag("task_type", task_type)
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("mode", mode)

        if is_custom:
            # Validated non-None above — df is None only for is_cv, which
            # never sets algorithm=custom (mutually exclusive branches).
            assert df is not None
            assert code_repo_url is not None
            assert entrypoint_path is not None
            mlflow.log_param("algorithm", "custom")
            mlflow.log_param("code_repo_url", code_repo_url)
            config = cast(dict[str, Any], json.loads(os.environ.get("CUSTOM_CONFIG") or "{}"))
            config["target_column"] = target_column
            with tempfile.TemporaryDirectory() as tmp_dir:
                model, metrics = run_custom_training(
                    df, config, code_repo_url, entrypoint_path, Path(tmp_dir) / "repo"
                )
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow_pyfunc.log_model(python_model=GenericPyfuncWrapper(model), artifact_path="model")
        elif architecture == "sklearn":
            # df/features are None only for is_cv, mutually exclusive here.
            assert df is not None
            assert features is not None
            spec = get_algorithm_spec(task_type, cast(str, algorithm))
            mlflow.log_param("algorithm", algorithm)
            if task_type == "clustering":
                # Transductive (no .predict) — clustering always fits+predicts on full data.
                if mode != "train":
                    raise RuntimeError("clustering does not support MODE=finetune")
                train_features, _ = _handle_missing_values(features, features, spec)
                train_features, _ = _scale_features(train_features, train_features, spec)
                model = spec.estimator_class()
                labels = model.fit_predict(train_features)
                metrics = compute_metrics(task_type, train_features, labels)
            elif task_type == "anomaly-detection":
                # Transductive, same as clustering — fits on the whole
                # dataset, never sees target_column even when one is set
                # (it's evaluation-only here, not a training signal).
                if mode != "train":
                    raise RuntimeError("anomaly-detection does not support MODE=finetune")
                train_features, _ = _handle_missing_values(features, features, spec)
                train_features, _ = _scale_features(train_features, train_features, spec)
                model = spec.estimator_class()
                raw_predictions = model.fit_predict(train_features)
                # sklearn's anomaly-estimator convention is -1=anomaly/
                # 1=normal — remapped to 1=anomaly/0=normal to match a
                # labeled column like is_anomaly, so compute_metrics'
                # precision/recall/f1 (when a label is provided) compare
                # like for like.
                predictions = np.where(raw_predictions == -1, 1, 0)
                true_labels = (
                    cast(pd.Series, df[target_column]) if target_column is not None else None
                )
                metrics = compute_metrics(task_type, true_labels, predictions)
            else:
                # Validated non-None above (only clustering/anomaly-detection,
                # handled in their own branches, allow it to be absent).
                assert target_column is not None
                labels_full = cast(pd.Series, df[target_column])
                train_features, test_features, train_labels, test_labels = _split(
                    df, features, labels_full, task_type, time_column
                )
                train_features, test_features = _handle_missing_values(
                    train_features, test_features, spec
                )
                train_features, test_features = _scale_features(train_features, test_features, spec)
                model = _fit(spec, mode, base_model_uri, train_features, train_labels)
                predictions = model.predict(test_features)
                metrics = compute_metrics(task_type, test_labels, predictions)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow_sklearn.log_model(model, artifact_path="model")
        elif is_nlp:
            # Validated non-None above (TEXT_COLUMN/TARGET_COLUMN both
            # required for ARCHITECTURE=nlp). df is None only for is_cv.
            assert df is not None
            assert text_column is not None
            assert target_column is not None
            # Raw df[[text_column]], not `features` — that's already category-encoded.
            text_features = cast(pd.DataFrame, df[[text_column]])
            labels_full = cast(pd.Series, df[target_column])
            train_features, test_features, train_labels, test_labels = _split(
                df, text_features, labels_full, task_type, time_column
            )
            hyperparameters = _read_nlp_hyperparameters()
            mlflow.log_param("base_model_name", hyperparameters["base_model_name"])
            model, metrics = train_nlp_and_evaluate(
                cast(pd.Series, train_features[text_column]),
                cast(pd.Series, test_features[text_column]),
                train_labels,
                test_labels,
                hyperparameters,
            )
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow_transformers.log_model(model, artifact_path="model")
        elif is_cv:
            # Same hyperparameter reader as DL — the 3 required keys are all
            # it reads when the DL-only optional env vars are unset.
            hyperparameters = _read_dl_hyperparameters()
            cv_model, metrics = train_cv_and_evaluate(dataset_path, hyperparameters)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow_pyfunc.log_model(
                python_model=GenericPyfuncWrapper(cv_model), artifact_path="model"
            )
        else:
            # target_column/df/features guaranteed set (is_cv can't reach this branch).
            assert df is not None
            assert features is not None
            assert target_column is not None
            labels_full = cast(pd.Series, df[target_column])
            train_features, test_features, train_labels, test_labels = _split(
                df, features, labels_full, task_type, time_column
            )
            hyperparameters = _read_dl_hyperparameters()
            if is_search:
                num_trials = int(os.environ["NUM_TRIALS"])
                search_space_config = json.loads(os.environ.get("SEARCH_SPACE_JSON") or "{}")
                objective_metric = os.environ["OBJECTIVE_METRIC"]
                objective_direction = os.environ.get("OBJECTIVE_DIRECTION", "maximize")
                strategy = build_search_strategy(search_strategy_name, objective_direction)
                spaces = build_search_spaces(search_space_config, hyperparameters)
                model, metrics, best_hyperparameters = run_hpo(
                    strategy,
                    hyperparameters,
                    spaces,
                    num_trials,
                    train_features,
                    test_features,
                    train_labels,
                    test_labels,
                    task_type,
                    architecture,
                    mode,
                    base_model_uri,
                    objective_metric,
                    objective_direction,
                )
                mlflow.log_param("search_strategy", search_strategy_name)
                mlflow.log_params({f"best_{k}": v for k, v in best_hyperparameters.items()})
            else:
                model, metrics = train_dl_and_evaluate(
                    train_features,
                    test_features,
                    train_labels,
                    test_labels,
                    task_type,
                    architecture,
                    hyperparameters,
                    mode,
                    base_model_uri,
                )
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            # serialization_format="pickle": mlflow's pytorch flavor now
            # defaults to "pt2" (torch.export-based), which requires an
            # input_example to trace the model graph — "pickle" matches the
            # sklearn/pyfunc flavors' behavior above, no example needed.
            mlflow_pytorch.log_model(model, artifact_path="model", serialization_format="pickle")

        if is_cv:
            # No pandas DataFrame to build an mlflow.data.Dataset from —
            # params carry the same lineage info instead.
            mlflow.log_param("dataset_uri", dataset_uri)
            mlflow.log_param("dataset_digest", dataset_digest)
        else:
            # mlflow.data's stub doesn't declare from_pandas even though
            # it's a real, documented function.
            dataset = mlflow_data.from_pandas(  # pyright: ignore[reportAttributeAccessIssue]
                df, source=dataset_uri, digest=dataset_digest
            )
            mlflow.log_input(dataset, context="training")
        artifact_uri = f"runs:/{run.info.run_id}/model"

    # Argo reads these back via outputs.parameters to hand off to register-step.
    Path("/tmp/artifact-uri").write_text(artifact_uri)
    Path("/tmp/dataset-digest").write_text(dataset_digest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — top-level: any failure must fail the Argo step, not hang.
        print(f"training failed: {exc}", file=sys.stderr)
        sys.exit(1)
