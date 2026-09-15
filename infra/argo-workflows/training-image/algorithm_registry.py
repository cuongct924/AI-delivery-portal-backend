"""Registry of algorithms Golden Path #1 (Train -> Track -> Register) can
train, keyed by task type then algorithm name — adding a new algorithm is
one dict entry, no other file changes needed.

Every estimator class exposes the standard scikit-learn `fit`/`predict`
interface (true for scikit-learn itself, and for XGBoost/LightGBM/CatBoost's
sklearn-compatible wrapper classes), so `train.py` can drive all of them
identically.
"""

from dataclasses import dataclass
from typing import Any, Final

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


@dataclass(frozen=True)
class AlgorithmSpec:
    """One registry entry.

    Attributes:
        estimator_class: scikit-learn-compatible class — must expose
            `fit`/`predict` (and `predict_proba`/no-op for clustering).
            Typed `type[Any]`, not `type[BaseEstimator]`: LightGBM/CatBoost's
            wrapper classes are duck-type compatible but don't actually
            subclass sklearn's BaseEstimator.
        requires_scaling: True for distance-/gradient-based estimators
            (KNN, SVC/SVR, clustering) where unscaled features skew results;
            False for tree-based estimators, which split on raw values.
        handles_missing_natively: True for the 3 boosting libraries — they
            learn an optimal split direction for NaN and can treat
            missingness itself as a signal. False means `train.py` must
            impute before fitting, or the estimator errors out.
    """

    estimator_class: type[Any]
    requires_scaling: bool
    handles_missing_natively: bool


_CLASSIFICATION: Final[dict[str, AlgorithmSpec]] = {
    "LogisticRegression": AlgorithmSpec(
        LogisticRegression, requires_scaling=True, handles_missing_natively=False
    ),
    "RandomForestClassifier": AlgorithmSpec(
        RandomForestClassifier, requires_scaling=False, handles_missing_natively=False
    ),
    "GradientBoostingClassifier": AlgorithmSpec(
        GradientBoostingClassifier, requires_scaling=False, handles_missing_natively=False
    ),
    "KNeighborsClassifier": AlgorithmSpec(
        KNeighborsClassifier, requires_scaling=True, handles_missing_natively=False
    ),
    "SVC": AlgorithmSpec(SVC, requires_scaling=True, handles_missing_natively=False),
    "GaussianNB": AlgorithmSpec(GaussianNB, requires_scaling=False, handles_missing_natively=False),
    "XGBClassifier": AlgorithmSpec(
        XGBClassifier, requires_scaling=False, handles_missing_natively=True
    ),
    "LGBMClassifier": AlgorithmSpec(
        LGBMClassifier, requires_scaling=False, handles_missing_natively=True
    ),
    "CatBoostClassifier": AlgorithmSpec(
        CatBoostClassifier, requires_scaling=False, handles_missing_natively=True
    ),
}

_REGRESSION: Final[dict[str, AlgorithmSpec]] = {
    "LinearRegression": AlgorithmSpec(
        LinearRegression, requires_scaling=True, handles_missing_natively=False
    ),
    "Ridge": AlgorithmSpec(Ridge, requires_scaling=True, handles_missing_natively=False),
    "Lasso": AlgorithmSpec(Lasso, requires_scaling=True, handles_missing_natively=False),
    "RandomForestRegressor": AlgorithmSpec(
        RandomForestRegressor, requires_scaling=False, handles_missing_natively=False
    ),
    "GradientBoostingRegressor": AlgorithmSpec(
        GradientBoostingRegressor, requires_scaling=False, handles_missing_natively=False
    ),
    "SVR": AlgorithmSpec(SVR, requires_scaling=True, handles_missing_natively=False),
    "XGBRegressor": AlgorithmSpec(
        XGBRegressor, requires_scaling=False, handles_missing_natively=True
    ),
    "LGBMRegressor": AlgorithmSpec(
        LGBMRegressor, requires_scaling=False, handles_missing_natively=True
    ),
    "CatBoostRegressor": AlgorithmSpec(
        CatBoostRegressor, requires_scaling=False, handles_missing_natively=True
    ),
}

# No XGBoost/LightGBM/CatBoost entry here — all 3 only ship supervised
# estimators, no clustering equivalent.
_CLUSTERING: Final[dict[str, AlgorithmSpec]] = {
    "KMeans": AlgorithmSpec(KMeans, requires_scaling=True, handles_missing_natively=False),
    "DBSCAN": AlgorithmSpec(DBSCAN, requires_scaling=True, handles_missing_natively=False),
    "AgglomerativeClustering": AlgorithmSpec(
        AgglomerativeClustering, requires_scaling=True, handles_missing_natively=False
    ),
    "GaussianMixture": AlgorithmSpec(
        GaussianMixture, requires_scaling=True, handles_missing_natively=False
    ),
}

# Transductive like clustering (train.py calls .fit_predict(), never
# .predict()) — LocalOutlierFactor's own .predict() only works with
# novelty=True, which this registry doesn't set, so fit_predict is the
# only method both estimators actually support here.
_ANOMALY_DETECTION: Final[dict[str, AlgorithmSpec]] = {
    "IsolationForest": AlgorithmSpec(
        IsolationForest, requires_scaling=False, handles_missing_natively=False
    ),
    "LocalOutlierFactor": AlgorithmSpec(
        LocalOutlierFactor, requires_scaling=True, handles_missing_natively=False
    ),
}

TASK_TYPE_ALGORITHMS: Final[dict[str, dict[str, AlgorithmSpec]]] = {
    "classification": _CLASSIFICATION,
    "regression": _REGRESSION,
    "clustering": _CLUSTERING,
    "anomaly-detection": _ANOMALY_DETECTION,
}


def get_algorithm_spec(task_type: str, algorithm: str) -> AlgorithmSpec:
    """Looks up a registry entry.

    Args:
        task_type: One of "classification", "regression", "clustering",
            "anomaly-detection".
        algorithm: Registry key, e.g. "XGBClassifier".

    Returns:
        The matching AlgorithmSpec.

    Raises:
        ValueError: task_type or algorithm isn't in the registry.
    """
    algorithms = TASK_TYPE_ALGORITHMS.get(task_type)
    if algorithms is None:
        raise ValueError(
            f"unknown task_type {task_type!r} — must be one of {sorted(TASK_TYPE_ALGORITHMS)}"
        )
    spec = algorithms.get(algorithm)
    if spec is None:
        raise ValueError(
            f"unknown algorithm {algorithm!r} for task_type {task_type!r} — "
            f"must be one of {sorted(algorithms)}"
        )
    return spec
