"""services/orchestration-api/data_quality/ — checks.py + registry.py."""

from typing import cast

import pandas as pd
from data_quality.checks import (
    check_class_imbalance,
    check_dimensionality_vs_samples,
    check_duplicate_rows,
    check_high_cardinality,
    check_missing_values,
    check_target_leakage_correlation,
    check_time_gaps,
)
from data_quality.registry import TASK_TYPE_CHECKS, UNIVERSAL_CHECKS, run_checks


def test_check_missing_values_blocks_on_missing_target() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, None, 0]})
    result = check_missing_values(df, target_column="y")
    assert result.severity == "blocking"


def test_check_missing_values_flags_mnar_correlation() -> None:
    # x is missing exactly when y == 1 — a textbook MNAR pattern.
    df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0, None], "y": [0, 1, 0, 1, 0, 1]})
    result = check_missing_values(df, target_column="y")
    assert result.severity == "warning"
    assert "x" in cast(dict, result.details["mnar_correlations"])


def test_check_missing_values_info_when_clean() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    result = check_missing_values(df, target_column="y")
    assert result.severity == "info"


def test_check_duplicate_rows_detects_duplicates() -> None:
    df = pd.DataFrame({"x": [1, 1, 2]})
    result = check_duplicate_rows(df)
    assert result.severity == "warning"
    assert result.details["duplicate_count"] == 1


def test_check_target_leakage_correlation_blocks_near_perfect_correlation() -> None:
    df = pd.DataFrame({"leak": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    result = check_target_leakage_correlation(df, target_column="y")
    assert result.severity == "blocking"


def test_check_target_leakage_correlation_skipped_without_target() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = check_target_leakage_correlation(df, target_column=None)
    assert result.severity == "info"


def test_check_class_imbalance_flags_rare_minority_class() -> None:
    df = pd.DataFrame({"y": [0] * 99 + [1]})
    result = check_class_imbalance(df, target_column="y")
    assert result.severity == "warning"


def test_check_high_cardinality_flags_near_unique_column() -> None:
    df = pd.DataFrame({"id_like": [f"v{i}" for i in range(200)]})
    result = check_high_cardinality(df)
    assert result.severity == "warning"
    assert "id_like" in cast(dict, result.details["cardinalities"])


def test_check_dimensionality_vs_samples_flags_more_features_than_rows() -> None:
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=pd.Index(list("abcde")))
    result = check_dimensionality_vs_samples(df)
    assert result.severity == "warning"


def test_check_time_gaps_flags_large_outlier_gap() -> None:
    df = pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-06-01"])}
    )
    result = check_time_gaps(df, time_column="t")
    assert result.severity == "warning"


def test_check_time_gaps_blocks_when_column_is_not_a_valid_date() -> None:
    # A column that's genuinely not a date (free text) — picking this as
    # Time column for architecture=lstm should fail loudly, not silently
    # read as "not enough timestamps to evaluate gaps".
    df = pd.DataFrame({"t": ["grocery", "electronics", "restaurant", "travel", "grocery"]})
    result = check_time_gaps(df, time_column="t")
    assert result.severity == "blocking"


def test_check_time_gaps_tolerates_a_few_unparseable_values() -> None:
    # Mostly-valid dates with a couple of bad rows shouldn't block —
    # unparseable_ratio stays under the threshold.
    df = pd.DataFrame(
        {
            "t": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "not-a-date",
            ]
        }
    )
    result = check_time_gaps(df, time_column="t")
    assert result.severity != "blocking"


def test_run_checks_includes_universal_and_task_type_checks() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    results = run_checks(df, "classification", target_column="y")
    names = {r.check_name for r in results}
    expected = {c.__name__ for c in UNIVERSAL_CHECKS} | {
        c.__name__ for c in TASK_TYPE_CHECKS["classification"]
    }
    assert names == expected


def test_run_checks_appends_time_gap_check_when_time_column_given() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1], "t": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    results = run_checks(df, "classification", target_column="y", time_column="t")
    assert "check_time_gaps" in {r.check_name for r in results}


def test_run_checks_clustering_ignores_missing_target_column() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    results = run_checks(df, "clustering")
    assert {r.check_name for r in results} == {
        "check_missing_values",
        "check_duplicate_rows",
        "check_dimensionality_vs_samples",
    }
