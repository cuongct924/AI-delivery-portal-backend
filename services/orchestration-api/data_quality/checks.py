"""Individual data quality checks — one pure function each, unit-testable in
isolation. registry.py decides which ones run for a given task type.

ETL validates schema ("does this load"); these checks validate ML fitness
("will this train a sane model") — a warehouse table can pass ETL and still
have leakage, class imbalance, or MNAR missingness that silently wrecks a
model, so this module runs independently of however the data got here.
"""

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

Severity = Literal["blocking", "warning", "info"]

# cast() below works around pandas' stubs over-widening `df[str]` to `Series | DataFrame`.

# Thresholds are simple fixed cutoffs, not tuned per dataset.
_HIGH_MISSING_RATIO = 0.3
_HIGH_CORRELATION = 0.9
_LEAKAGE_CORRELATION = 0.98
_MNAR_CORRELATION = 0.3
_MINORITY_CLASS_RATIO = 0.05
_HIGH_CARDINALITY_RATIO = 0.5
_HIGH_CARDINALITY_ABSOLUTE = 100
_GAP_OUTLIER_MULTIPLIER = 10
_UNPARSEABLE_TIME_RATIO = 0.2


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    severity: Severity
    message: str
    details: dict[str, object]


def check_missing_values(df: pd.DataFrame, target_column: str | None = None) -> CheckResult:
    """Flags missing values — blocking only when the label itself is
    incomplete (can't supervise-train on a missing target); otherwise
    reports per-column ratios and, if the missingness pattern correlates
    with the target, calls that out as a signal worth keeping rather than
    naively imputing away (a column that's more often missing for one
    outcome than another is itself predictive)."""
    if target_column is not None and target_column in df.columns:
        target_series = cast(pd.Series, df[target_column])
        if target_series.isna().any():
            missing_count = int(target_series.isna().sum())
            return CheckResult(
                "check_missing_values",
                "blocking",
                f"target column {target_column!r} has {missing_count} missing value(s) — "
                "cannot train on missing labels",
                {"target_missing_count": missing_count},
            )
    else:
        target_series = None

    ratios = cast(pd.Series, df.isna().mean()).to_dict()
    high_missing = {col: r for col, r in ratios.items() if r > _HIGH_MISSING_RATIO}

    mnar_signals: dict[str, float] = {}
    if target_series is not None and pd.api.types.is_numeric_dtype(target_series):
        for column in df.columns:
            if column == target_column or df[column].isna().sum() == 0:
                continue
            # pandas' bundled stub resolves corr()'s overload/return type
            # unreliably — cast to the actual runtime type (float).
            correlation = cast(
                float,
                df[column].isna().astype(int).corr(target_series),  # pyright: ignore
            )
            if pd.notna(correlation) and abs(correlation) >= _MNAR_CORRELATION:
                mnar_signals[column] = round(correlation, 3)

    if mnar_signals:
        return CheckResult(
            "check_missing_values",
            "warning",
            f"missingness in {list(mnar_signals)} correlates with the target — "
            "consider keeping as a signal instead of imputing",
            {"missing_ratios": ratios, "mnar_correlations": mnar_signals},
        )
    if high_missing:
        return CheckResult(
            "check_missing_values",
            "warning",
            f"columns with >{_HIGH_MISSING_RATIO:.0%} missing values: {list(high_missing)}",
            {"missing_ratios": ratios},
        )
    return CheckResult(
        "check_missing_values",
        "info",
        "no columns exceed the missing-value warning threshold",
        {"missing_ratios": ratios},
    )


def check_duplicate_rows(df: pd.DataFrame, target_column: str | None = None) -> CheckResult:
    # target_column accepted-but-unused so registry.run_checks can call every
    # check with the same (df, target_column=...) signature.
    del target_column
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        return CheckResult(
            "check_duplicate_rows",
            "warning",
            f"{duplicate_count} duplicate row(s) found",
            {"duplicate_count": duplicate_count},
        )
    return CheckResult(
        "check_duplicate_rows", "info", "no duplicate rows found", {"duplicate_count": 0}
    )


def check_target_leakage_correlation(
    df: pd.DataFrame, target_column: str | None = None
) -> CheckResult:
    """Flags features near-perfectly correlated with the target — usually a
    sign the feature encodes the label itself (leakage), not a genuinely
    predictive relationship."""
    if target_column is None:
        return CheckResult(
            "check_target_leakage_correlation",
            "info",
            "no target_column provided — check skipped",
            {},
        )
    target_series = cast(pd.Series, df[target_column])
    if not pd.api.types.is_numeric_dtype(target_series):
        return CheckResult(
            "check_target_leakage_correlation",
            "info",
            "target is non-numeric — correlation check skipped",
            {},
        )

    correlations: dict[str, float] = {}
    for column in df.columns:
        if column == target_column or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        # pandas' bundled stub resolves corr()'s overload/return type
        # unreliably — cast to the actual runtime type (float).
        correlation = cast(
            float,
            df[column].corr(target_series),  # pyright: ignore
        )
        if pd.notna(correlation):
            correlations[column] = round(correlation, 3)

    leaking = {c: v for c, v in correlations.items() if abs(v) >= _LEAKAGE_CORRELATION}
    if leaking:
        return CheckResult(
            "check_target_leakage_correlation",
            "blocking",
            f"feature(s) {list(leaking)} are near-perfectly correlated with the target — "
            "likely leakage",
            {"correlations": leaking},
        )
    suspicious = {c: v for c, v in correlations.items() if abs(v) >= _HIGH_CORRELATION}
    if suspicious:
        return CheckResult(
            "check_target_leakage_correlation",
            "warning",
            f"feature(s) {list(suspicious)} are highly correlated with the target — "
            "worth a manual check",
            {"correlations": suspicious},
        )
    return CheckResult(
        "check_target_leakage_correlation",
        "info",
        "no suspiciously high target correlations found",
        {"correlations": correlations},
    )


def check_class_imbalance(df: pd.DataFrame, target_column: str | None = None) -> CheckResult:
    if target_column is None:
        return CheckResult(
            "check_class_imbalance", "info", "no target_column provided — check skipped", {}
        )
    counts = df[target_column].value_counts()
    if counts.empty:
        return CheckResult("check_class_imbalance", "info", "no rows to evaluate class balance", {})
    ratios = (counts / counts.sum()).to_dict()
    minority_ratio = min(ratios.values())
    if minority_ratio < _MINORITY_CLASS_RATIO:
        return CheckResult(
            "check_class_imbalance",
            "warning",
            f"minority class is only {minority_ratio:.1%} of rows — "
            "consider class weighting or resampling",
            {"class_ratios": ratios},
        )
    return CheckResult(
        "check_class_imbalance",
        "info",
        "class distribution is reasonably balanced",
        {"class_ratios": ratios},
    )


def check_high_cardinality(df: pd.DataFrame, target_column: str | None = None) -> CheckResult:
    """High-cardinality categorical columns break the assumption behind
    automatic ordinal encoding — near-unique values encode to near-unique
    codes, giving the model no generalizable signal."""
    del target_column  # unused — see check_duplicate_rows
    row_count = len(df)
    flagged: dict[str, int] = {}
    for column in df.select_dtypes(include="object").columns:
        unique_count = df[column].nunique()
        if unique_count > _HIGH_CARDINALITY_ABSOLUTE or (
            row_count > 0 and unique_count / row_count > _HIGH_CARDINALITY_RATIO
        ):
            flagged[column] = int(unique_count)
    if flagged:
        return CheckResult(
            "check_high_cardinality",
            "warning",
            f"high-cardinality column(s): {flagged}",
            {"cardinalities": flagged},
        )
    return CheckResult("check_high_cardinality", "info", "no high-cardinality columns found", {})


def check_dimensionality_vs_samples(
    df: pd.DataFrame, target_column: str | None = None
) -> CheckResult:
    """Clustering-specific — more features than samples makes distance
    metrics unreliable (curse of dimensionality)."""
    del target_column  # unused — see check_duplicate_rows
    feature_count = df.shape[1]
    sample_count = df.shape[0]
    if sample_count > 0 and feature_count > sample_count:
        return CheckResult(
            "check_dimensionality_vs_samples",
            "warning",
            f"{feature_count} features but only {sample_count} samples — "
            "distances become unreliable in high dimensions",
            {"feature_count": feature_count, "sample_count": sample_count},
        )
    return CheckResult(
        "check_dimensionality_vs_samples",
        "info",
        "feature count is reasonable relative to sample count",
        {"feature_count": feature_count, "sample_count": sample_count},
    )


def check_time_gaps(df: pd.DataFrame, time_column: str) -> CheckResult:
    """Large gaps in a time-ordered column can mean missing periods —
    relevant to LSTM windowing and to TimeSeriesSplit's assumption of
    reasonably even coverage.

    Also the only check that verifies `time_column` is actually a date/time
    column at all: `pd.to_datetime(errors="coerce")` silently turns
    anything it can't parse into NaT, so a column picked by mistake (wrong
    dtype, free text) would otherwise just fall through to "not enough
    timestamps to evaluate gaps" at `info` severity — indistinguishable
    from a genuinely tiny but valid dataset. Blocking here instead is what
    actually catches "this architecture=lstm run has no real date column."
    """
    raw = df[time_column]
    parsed = pd.to_datetime(raw, errors="coerce")
    unparseable_ratio = cast(float, parsed.isna().mean()) if len(raw) > 0 else 0.0
    if unparseable_ratio > _UNPARSEABLE_TIME_RATIO:
        return CheckResult(
            "check_time_gaps",
            "blocking",
            f"{time_column!r} isn't a valid date/time column — "
            f"{unparseable_ratio:.0%} of values don't parse as a date",
            {"unparseable_ratio": round(unparseable_ratio, 3)},
        )
    timestamps = parsed.dropna().sort_values()
    if len(timestamps) < 3:
        return CheckResult("check_time_gaps", "info", "not enough timestamps to evaluate gaps", {})
    gaps = timestamps.diff().dropna()
    # Series.median()'s stub isn't dtype-specialized — it's actually a
    # Timedelta here (gaps come from diffing datetimes), not a float.
    median_gap = cast(pd.Timedelta, gaps.median())
    if median_gap == pd.Timedelta(0):
        return CheckResult(
            "check_time_gaps",
            "info",
            "median gap between timestamps is zero — skipping outlier check",
            {},
        )
    outliers = cast(pd.Series, gaps[gaps > median_gap * _GAP_OUTLIER_MULTIPLIER])
    if not outliers.empty:
        return CheckResult(
            "check_time_gaps",
            "warning",
            f"{len(outliers)} gap(s) more than {_GAP_OUTLIER_MULTIPLIER}x the median — "
            "possible missing time periods",
            {
                "outlier_gap_count": int(len(outliers)),
                "median_gap_seconds": median_gap.total_seconds(),
            },
        )
    return CheckResult(
        "check_time_gaps",
        "info",
        "no unusually large gaps found",
        {"median_gap_seconds": median_gap.total_seconds()},
    )


# RecSys checks don't fit run_checks()'s (df, target_column) shape — called
# directly by routers/recommendations.py instead.
_MIN_INTERACTIONS_PER_ENTITY = 5  # k-core threshold for "not cold-start"


def check_rec_ids_present(
    interactions: pd.DataFrame, user_id_column: str, item_id_column: str
) -> CheckResult:
    """Flags missing id columns or null ids — a null user/item id can't be
    trained or evaluated on at all, unlike an ordinary missing feature."""
    missing_columns = [c for c in (user_id_column, item_id_column) if c not in interactions.columns]
    if missing_columns:
        return CheckResult(
            "check_rec_ids_present",
            "blocking",
            f"missing column(s): {missing_columns}",
            {"missing_columns": missing_columns},
        )
    null_users = int(cast(pd.Series, interactions[user_id_column]).isna().sum())
    null_items = int(cast(pd.Series, interactions[item_id_column]).isna().sum())
    if null_users or null_items:
        return CheckResult(
            "check_rec_ids_present",
            "blocking",
            f"{null_users} null user id(s), {null_items} null item id(s)",
            {"null_user_ids": null_users, "null_item_ids": null_items},
        )
    return CheckResult("check_rec_ids_present", "info", "no missing id columns or null ids", {})


def check_rec_duplicate_interactions(
    interactions: pd.DataFrame, user_id_column: str, item_id_column: str
) -> CheckResult:
    """Flags duplicate (user, item) pairs — usually a sign the same
    interaction got logged more than once upstream, which would silently
    overweight it during training."""
    duplicate_count = int(interactions.duplicated(subset=[user_id_column, item_id_column]).sum())
    if duplicate_count > 0:
        return CheckResult(
            "check_rec_duplicate_interactions",
            "warning",
            f"{duplicate_count} duplicate (user, item) pair(s)",
            {"duplicate_count": duplicate_count},
        )
    return CheckResult("check_rec_duplicate_interactions", "info", "no duplicate interactions", {})


def check_rec_cold_start_ratio(
    interactions: pd.DataFrame, user_id_column: str, item_id_column: str
) -> CheckResult:
    """Warns when a large share of users/items fall below the k-core
    threshold — too little interaction history for any algorithm family
    to learn a meaningful signal for them, regardless of which one Dev
    picks."""
    user_counts = interactions[user_id_column].value_counts()
    item_counts = interactions[item_id_column].value_counts()
    cold_user_ratio = (
        (user_counts < _MIN_INTERACTIONS_PER_ENTITY).mean() if len(user_counts) else 0.0
    )
    cold_item_ratio = (
        (item_counts < _MIN_INTERACTIONS_PER_ENTITY).mean() if len(item_counts) else 0.0
    )
    details = {
        "cold_user_ratio": float(cold_user_ratio),
        "cold_item_ratio": float(cold_item_ratio),
        "k_core_threshold": _MIN_INTERACTIONS_PER_ENTITY,
    }
    if cold_user_ratio > 0.5 or cold_item_ratio > 0.5:
        return CheckResult(
            "check_rec_cold_start_ratio",
            "warning",
            f"{cold_user_ratio:.0%} of users and {cold_item_ratio:.0%} of items have fewer than "
            f"{_MIN_INTERACTIONS_PER_ENTITY} interactions — expect weak recommendations for them",
            details,
        )
    return CheckResult(
        "check_rec_cold_start_ratio", "info", "cold-start ratio within range", details
    )
