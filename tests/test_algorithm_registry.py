"""infra/argo-workflows/training-image/algorithm_registry.py — registry
lookup dispatch, plus a light sanity check that every registered class is
actually instantiable and exposes fit/predict."""

import pytest
from algorithm_registry import TASK_TYPE_ALGORITHMS, get_algorithm_spec


@pytest.mark.parametrize(
    "task_type", ["classification", "regression", "clustering", "anomaly-detection"]
)
def test_every_registered_estimator_exposes_fit_and_predict(task_type: str) -> None:
    for name, spec in TASK_TYPE_ALGORITHMS[task_type].items():
        estimator = spec.estimator_class()
        assert hasattr(estimator, "fit"), f"{name} has no fit()"
        # DBSCAN/AgglomerativeClustering are transductive — fit_predict
        # instead of predict, train.py branches on task_type for this.
        assert hasattr(estimator, "predict") or hasattr(estimator, "fit_predict"), (
            f"{name} has no predict()/fit_predict()"
        )


def test_no_boosting_library_entries_in_clustering() -> None:
    # XGBoost/LightGBM/CatBoost only ship supervised estimators.
    clustering_classes = {
        spec.estimator_class.__module__.split(".")[0]
        for spec in TASK_TYPE_ALGORITHMS["clustering"].values()
    }
    assert clustering_classes == {"sklearn"}


def test_no_boosting_library_entries_in_anomaly_detection() -> None:
    anomaly_classes = {
        spec.estimator_class.__module__.split(".")[0]
        for spec in TASK_TYPE_ALGORITHMS["anomaly-detection"].values()
    }
    assert anomaly_classes == {"sklearn"}


def test_get_algorithm_spec_returns_matching_entry() -> None:
    spec = get_algorithm_spec("classification", "XGBClassifier")
    assert spec.estimator_class.__name__ == "XGBClassifier"
    assert spec.handles_missing_natively is True
    assert spec.requires_scaling is False


def test_get_algorithm_spec_rejects_unknown_task_type() -> None:
    with pytest.raises(ValueError, match="unknown task_type"):
        get_algorithm_spec("not-a-task-type", "XGBClassifier")


def test_get_algorithm_spec_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unknown algorithm"):
        get_algorithm_spec("classification", "NotARealAlgorithm")
