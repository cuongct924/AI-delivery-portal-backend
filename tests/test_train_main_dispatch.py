"""infra/argo-workflows/training-image/train.py — main()'s ARCHITECTURE
dispatch. Mocks mlflow (module + submodule imports) and train_dl's
train_and_evaluate so this runs with no MLflow server and no real training,
exercising only the branching logic itself.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_BASE_ENV = {
    "TASK_TYPE": "regression",
    "TARGET_COLUMN": "target",
    "ID_COLUMNS": "",
    "MODE": "train",
    "BASE_MODEL_URI": "",
    "TIME_COLUMN": "",
}


def _write_dataset(tmp_path: Path) -> Path:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df.to_csv(csv_path, index=False)
    (tmp_path / "data.csv.dvc").write_text("outs:\n- md5: deadbeef\n  path: data.csv\n")
    return csv_path


def _set_env(
    monkeypatch: pytest.MonkeyPatch, csv_path: Path, tmp_path: Path, **overrides: str
) -> None:
    env = {**_BASE_ENV, "DATASET_URI": f"file://{csv_path}", **overrides}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ARCHITECTURE", raising=False)
    monkeypatch.chdir(tmp_path)


@patch("train.mlflow_sklearn")
@patch("train.mlflow_data")
@patch("train.mlflow")
def test_main_sklearn_branch_unchanged_when_architecture_unset(
    mock_mlflow: MagicMock,
    mock_mlflow_data: MagicMock,
    mock_mlflow_sklearn: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_dataset(tmp_path)
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-1"
    mock_dl = MagicMock(side_effect=AssertionError("DL path must not run for architecture=sklearn"))
    monkeypatch.setattr("train.train_dl_and_evaluate", mock_dl)
    _set_env(monkeypatch, csv_path, tmp_path, ALGORITHM="LinearRegression")

    import train

    train.main()

    mock_mlflow_sklearn.log_model.assert_called_once()
    mock_dl.assert_not_called()


@patch("train.mlflow_pytorch")
@patch("train.mlflow_data")
@patch("train.mlflow")
def test_main_dl_branch_dispatches_to_train_dl_when_architecture_set(
    mock_mlflow: MagicMock,
    mock_mlflow_data: MagicMock,
    mock_mlflow_pytorch: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = _write_dataset(tmp_path)
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-2"
    fake_model = MagicMock()
    mock_dl = MagicMock(
        return_value=(fake_model, {"r2": 0.9, "mean_absolute_percentage_error": 0.1})
    )
    monkeypatch.setattr("train.train_dl_and_evaluate", mock_dl)
    _set_env(
        monkeypatch,
        csv_path,
        tmp_path,
        LEARNING_RATE="0.01",
        EPOCHS="1",
        BATCH_SIZE="4",
        HIDDEN_LAYERS="8,4",
        DROPOUT="0.0",
        OPTIMIZER="sgd",
    )
    monkeypatch.setenv("ARCHITECTURE", "mlp")

    import train

    train.main()

    mock_dl.assert_called_once()
    called_args = mock_dl.call_args.args
    assert called_args[4] == "regression"  # task_type
    assert called_args[5] == "mlp"  # architecture
    assert called_args[6]["optimizer"] == "sgd"  # hyperparameters
    mock_mlflow_pytorch.log_model.assert_called_once_with(
        fake_model, artifact_path="model", serialization_format="pickle"
    )


def test_main_cv_branch_dispatches_to_train_cv_and_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        patch("train.mlflow_pyfunc") as mock_mlflow_pyfunc,
        patch("train.mlflow") as mock_mlflow,
    ):
        zip_path = tmp_path / "shapes.zip"
        zip_path.write_bytes(b"fake zip bytes, train_cv_and_evaluate is mocked, never reads it")
        (tmp_path / "shapes.zip.dvc").write_text("outs:\n- md5: deadbeef\n  path: shapes.zip\n")
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-cv"
        fake_model = MagicMock()
        mock_train_cv = MagicMock(return_value=(fake_model, {"accuracy": 0.7}))
        monkeypatch.setattr("train.train_cv_and_evaluate", mock_train_cv)
        env = {
            "DATASET_URI": f"file://{zip_path}",
            "TASK_TYPE": "classification",
            "MODE": "train",
            "LEARNING_RATE": "0.01",
            "EPOCHS": "1",
            "BATCH_SIZE": "4",
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("ARCHITECTURE", "cv")
        monkeypatch.chdir(tmp_path)

        import train

        train.main()

        mock_train_cv.assert_called_once_with(
            zip_path, {"learning_rate": 0.01, "epochs": 1, "batch_size": 4}
        )
        mock_mlflow_pyfunc.log_model.assert_called_once()
        assert mock_mlflow_pyfunc.log_model.call_args.kwargs["artifact_path"] == "model"
        mock_mlflow.log_metric.assert_any_call("accuracy", 0.7)
        mock_mlflow.log_param.assert_any_call("dataset_uri", f"file://{zip_path}")


def test_main_cv_rejects_non_classification_task_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "shapes.zip"
    zip_path.write_bytes(b"irrelevant, validated before any read")
    monkeypatch.setenv("DATASET_URI", f"file://{zip_path}")
    monkeypatch.setenv("TASK_TYPE", "regression")
    monkeypatch.setenv("MODE", "train")
    monkeypatch.setenv("ARCHITECTURE", "cv")

    import train

    with pytest.raises(RuntimeError, match="only supports TASK_TYPE=classification"):
        train.main()


def test_main_nlp_branch_dispatches_to_train_nlp_and_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        patch("train.mlflow_transformers") as mock_mlflow_transformers,
        patch("train.mlflow_data"),
        patch("train.mlflow") as mock_mlflow,
    ):
        csv_path = tmp_path / "reviews.csv"
        pd.DataFrame(
            {
                "review": ["great", "bad", "great", "bad", "great", "bad"],
                "sentiment": ["pos", "neg", "pos", "neg", "pos", "neg"],
            }
        ).to_csv(csv_path, index=False)
        (tmp_path / "reviews.csv.dvc").write_text("outs:\n- md5: deadbeef\n  path: reviews.csv\n")
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-nlp"
        fake_model = {"model": MagicMock(), "tokenizer": MagicMock()}
        mock_train_nlp = MagicMock(return_value=(fake_model, {"accuracy": 0.8}))
        monkeypatch.setattr("train.train_nlp_and_evaluate", mock_train_nlp)
        env = {
            **_BASE_ENV,
            "DATASET_URI": f"file://{csv_path}",
            "TASK_TYPE": "classification",
            "TARGET_COLUMN": "sentiment",
            "TEXT_COLUMN": "review",
            "BASE_MODEL_NAME": "distilbert-base-uncased",
            "LEARNING_RATE": "0.001",
            "EPOCHS": "1",
            "BATCH_SIZE": "2",
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("ARCHITECTURE", "nlp")
        monkeypatch.chdir(tmp_path)

        import train

        train.main()

        mock_train_nlp.assert_called_once()
        call_args = mock_train_nlp.call_args.args
        train_text, test_text = call_args[0], call_args[1]
        assert set(train_text).issubset({"great", "bad"})
        assert len(train_text) + len(test_text) == 6
        mock_mlflow_transformers.log_model.assert_called_once_with(
            fake_model, artifact_path="model"
        )
        mock_mlflow.log_metric.assert_any_call("accuracy", 0.8)


def test_main_nlp_requires_text_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = _write_dataset(tmp_path)
    _set_env(monkeypatch, csv_path, tmp_path, TASK_TYPE="classification")
    monkeypatch.setenv("ARCHITECTURE", "nlp")

    import train

    with pytest.raises(RuntimeError, match="TEXT_COLUMN is required"):
        train.main()


def test_main_nlp_rejects_non_classification_task_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _write_dataset(tmp_path)
    _set_env(monkeypatch, csv_path, tmp_path, TEXT_COLUMN="f1")
    monkeypatch.setenv("ARCHITECTURE", "nlp")

    import train

    with pytest.raises(RuntimeError, match="only supports TASK_TYPE=classification"):
        train.main()


def test_main_hpo_branch_dispatches_to_run_hpo_when_search_strategy_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        patch("train.mlflow_pytorch") as mock_mlflow_pytorch,
        patch("train.mlflow_data"),
        patch("train.mlflow") as mock_mlflow,
    ):
        csv_path = _write_dataset(tmp_path)
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-hpo"
        fake_model = MagicMock()
        mock_run_hpo = MagicMock(
            return_value=(fake_model, {"r2": 0.95}, {"learning_rate": 0.01, "epochs": 10})
        )
        monkeypatch.setattr("train.run_hpo", mock_run_hpo)
        monkeypatch.setattr("train.train_dl_and_evaluate", MagicMock(side_effect=AssertionError))
        _set_env(
            monkeypatch,
            csv_path,
            tmp_path,
            LEARNING_RATE="0.01",
            EPOCHS="10",
            BATCH_SIZE="4",
            HIDDEN_LAYERS="8,4",
            DROPOUT="0.0",
            SEARCH_STRATEGY="bayesian",
            NUM_TRIALS="5",
            SEARCH_SPACE_JSON='{"learning_rate": {"low": 0.001, "high": 0.1}}',
            OBJECTIVE_METRIC="r2",
            OBJECTIVE_DIRECTION="maximize",
        )
        monkeypatch.setenv("ARCHITECTURE", "mlp")

        import train

        train.main()

        mock_run_hpo.assert_called_once()
        mock_mlflow_pytorch.log_model.assert_called_once_with(
            fake_model, artifact_path="model", serialization_format="pickle"
        )
        mock_mlflow.log_param.assert_any_call("search_strategy", "bayesian")


def test_main_rejects_search_strategy_for_sklearn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _write_dataset(tmp_path)
    _set_env(monkeypatch, csv_path, tmp_path, ALGORITHM="LinearRegression", SEARCH_STRATEGY="grid")

    import train

    with pytest.raises(RuntimeError, match="SEARCH_STRATEGY != 'fixed' requires ARCHITECTURE"):
        train.main()


def test_main_requires_algorithm_when_architecture_is_sklearn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _write_dataset(tmp_path)
    _set_env(monkeypatch, csv_path, tmp_path)  # no ALGORITHM set

    import train

    with pytest.raises(RuntimeError, match="ALGORITHM is required"):
        train.main()


def test_main_rejects_dl_architecture_for_clustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _write_dataset(tmp_path)
    _set_env(monkeypatch, csv_path, tmp_path, TASK_TYPE="clustering", TARGET_COLUMN="")
    monkeypatch.setenv("ARCHITECTURE", "lstm")

    import train

    with pytest.raises(RuntimeError, match="does not support task_type='clustering'"):
        train.main()


def test_main_rejects_dl_architecture_for_anomaly_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _write_dataset(tmp_path)
    _set_env(monkeypatch, csv_path, tmp_path, TASK_TYPE="anomaly-detection", TARGET_COLUMN="")
    monkeypatch.setenv("ARCHITECTURE", "lstm")

    import train

    with pytest.raises(RuntimeError, match="does not support task_type='anomaly-detection'"):
        train.main()


@patch("train.mlflow_sklearn")
@patch("train.mlflow_data")
@patch("train.mlflow")
def test_main_anomaly_detection_runs_without_a_target_column(
    mock_mlflow: MagicMock,
    mock_mlflow_data: MagicMock,
    mock_mlflow_sklearn: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No TARGET_COLUMN at all — purely unsupervised, unlike
    # classification/regression which require one (see the guard this
    # exercises: task_type not in ("clustering", "anomaly-detection")).
    csv_path = _write_dataset(tmp_path)
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-anomaly"
    _set_env(
        monkeypatch,
        csv_path,
        tmp_path,
        TASK_TYPE="anomaly-detection",
        TARGET_COLUMN="",
        ALGORITHM="IsolationForest",
    )

    import train

    train.main()

    mock_mlflow_sklearn.log_model.assert_called_once()
    logged_metric_names = {call.args[0] for call in mock_mlflow.log_metric.call_args_list}
    assert logged_metric_names == {"anomaly_rate"}
