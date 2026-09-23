"""Individual data quality checks — one pure function each, unit-testable in
isolation. registry.py decides which ones run for a given task type.

ETL validates schema ("does this load"); these checks validate ML fitness
("will this train a sane model") — a warehouse table can pass ETL and still
have leakage, class imbalance, or MNAR missingness that silently wrecks a
model, so this module runs independently of however the data got here.

Column/Series/shape-level thresholds (target nullability, high missing
ratio, high cardinality, class balance, dimensionality, date parseability)
are expressed as Pandera schemas — Pandera already owns "does this column/
series/dataframe satisfy property X", with standardized, structured failure
reporting (`SchemaErrors.failure_cases`) instead of a hand-rolled dict
comprehension per check. Every custom `Check` below passes `ignore_na=False`
— Pandera's default (`ignore_na=True`) drops null values from the Series
*before* calling the check function, which silently breaks any check whose
own predicate is about the null ratio itself (missing-value checks would
always see a NaN-free Series and never fire).

Cross-column statistical relationships (target-leakage correlation, MNAR
missingness correlation) stay plain pandas: a Pandera Check only ever sees
the one column/series/dataframe it's attached to, and both of these need the
*actual correlation values* back (for the message and for a 2-tier
blocking/warning threshold), not just a pass/fail — smuggling a second
column into the check via closure would satisfy the type checker but buys
nothing over calling pandas directly, since the values still have to be
extracted separately either way. Whole-row duplicate detection is a similar
case: Pandera's composite-uniqueness check flags one failure_cases row per
*column* per duplicate occurrence, not one per duplicate row, so recovering
the same "N duplicate rows" count pandas' `.duplicated().sum()` gives for
free would mean re-deriving it from pandas anyway.
"""

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd
import pandera.pandas as pa

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


def _flagged_columns(errors: pa.errors.SchemaErrors) -> list[str]:
    """Distinct column names Pandera flagged, in first-seen order — a
    coerce failure can emit more than one failure_cases row per column."""
    return list(dict.fromkeys(c for c in errors.failure_cases["column"] if c is not None))


def _flagged_row_count(errors: pa.errors.SchemaErrors) -> int:
    """Distinct row count Pandera flagged — a coerce failure emits one
    failure_cases row per check it fails against the same value (e.g. both
    `coerce_dtype` and the resulting `dtype` check), so counting distinct
    `index` values avoids double-counting the same bad row."""
    return int(errors.failure_cases["index"].nunique())


def check_missing_values(df: pd.DataFrame, target_column: str | None = None) -> CheckResult:
    """Flags missing values — blocking only when the label itself is
    incomplete (can't supervise-train on a missing target); otherwise
    reports per-column ratios and, if the missingness pattern correlates
    with the target, calls that out as a signal worth keeping rather than
    naively imputing away (a column that's more often missing for one
    outcome than another is itself predictive)."""
    target_series = None
    if target_column is not None and target_column in df.columns:
        target_series = cast(pd.Series, df[target_column])
        target_not_null_schema = pa.DataFrameSchema({target_column: pa.Column(nullable=False)})
        try:
            # df[[col]] is a DataFrame at runtime — pandas' stub over-widens
            # list-of-one column indexing the same way it over-widens df[str].
            target_not_null_schema.validate(cast(pd.DataFrame, df[[target_column]]), lazy=True)
        except pa.errors.SchemaErrors as errors:
            missing_count = _flagged_row_count(errors)
            return CheckResult(
                "check_missing_values",
                "blocking",
                f"target column {target_column!r} has {missing_count} missing value(s) — "
                "cannot train on missing labels",
                {"target_missing_count": missing_count},
            )

    ratios = cast(pd.Series, df.isna().mean()).to_dict()

    high_missing_schema = pa.DataFrameSchema(
        {
            column: pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda s: not (s.isna().mean() > _HIGH_MISSING_RATIO),
                    error="high_missing_ratio",
                    ignore_na=False,
                ),
            )
            for column in df.columns
        }
    )
    try:
        high_missing_schema.validate(df, lazy=True)
        high_missing_columns: list[str] = []
    except pa.errors.SchemaErrors as errors:
        high_missing_columns = _flagged_columns(errors)

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
    if high_missing_columns:
        return CheckResult(
            "check_missing_values",
            "warning",
            f"columns with >{_HIGH_MISSING_RATIO:.0%} missing values: {high_missing_columns}",
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
    schema = pa.DataFrameSchema(
        checks=pa.Check(lambda d: not d.duplicated().any(), error="duplicate_rows", ignore_na=False)
    )
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors:
        # Pandera's own composite-uniqueness check reports one failure_cases
        # row per column per duplicate occurrence, not one per duplicate
        # row — recomputing the count via pandas directly is simpler and
        # exactly matches what the message/details need.
        duplicate_count = int(df.duplicated().sum())
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

    balance_schema = pa.SeriesSchema(
        checks=pa.Check(
            lambda s: not (s.value_counts(normalize=True).min() < _MINORITY_CLASS_RATIO),
            error="minority_class_too_small",
            ignore_na=False,
        )
    )
    try:
        balance_schema.validate(cast(pd.Series, df[target_column]), lazy=True)
    except pa.errors.SchemaErrors:
        minority_ratio = min(ratios.values())
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
    object_columns = df.select_dtypes(include="object").columns
    row_count = len(df)

    def _is_high_cardinality(s: pd.Series) -> bool:
        unique_count = s.nunique()
        return not (
            unique_count > _HIGH_CARDINALITY_ABSOLUTE
            or (row_count > 0 and unique_count / row_count > _HIGH_CARDINALITY_RATIO)
        )

    flagged_columns: list[str] = []
    if len(object_columns) > 0:
        schema = pa.DataFrameSchema(
            {
                column: pa.Column(
                    checks=pa.Check(_is_high_cardinality, error="high_cardinality", ignore_na=False)
                )
                for column in object_columns
            }
        )
        try:
            schema.validate(cast(pd.DataFrame, df[object_columns]), lazy=True)
        except pa.errors.SchemaErrors as errors:
            flagged_columns = _flagged_columns(errors)

    if flagged_columns:
        cardinalities = {column: int(df[column].nunique()) for column in flagged_columns}
        return CheckResult(
            "check_high_cardinality",
            "warning",
            f"high-cardinality column(s): {cardinalities}",
            {"cardinalities": cardinalities},
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

    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: not (d.shape[0] > 0 and d.shape[1] > d.shape[0]),
            error="too_many_features",
            ignore_na=False,
        )
    )
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors:
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
    column at all: Pandera's own `coerce=True` runs the same
    `pd.to_datetime`-style parsing pandas did directly before, and reports
    exactly which rows failed to coerce — a column picked by mistake
    (wrong dtype, free text) surfaces as a blocking failure here instead of
    silently falling through to "not enough timestamps to evaluate gaps" at
    `info` severity, indistinguishable from a genuinely tiny but valid
    dataset.
    """
    raw = df[time_column]
    schema = pa.DataFrameSchema(
        {time_column: pa.Column("datetime64[ns]", coerce=True, nullable=True)}
    )
    try:
        parsed_df = schema.validate(cast(pd.DataFrame, df[[time_column]]), lazy=True)
        unparseable_ratio = 0.0
    except pa.errors.SchemaErrors as errors:
        unparseable_ratio = _flagged_row_count(errors) / len(raw) if len(raw) > 0 else 0.0
        # Coercion still fills in what it could parse — pull it out of the
        # partially-coerced object Pandera hands back on failure, same as
        # `pd.to_datetime(raw, errors="coerce")` used to.
        parsed_df = pd.DataFrame({time_column: pd.to_datetime(raw, errors="coerce")})

    if unparseable_ratio > _UNPARSEABLE_TIME_RATIO:
        return CheckResult(
            "check_time_gaps",
            "blocking",
            f"{time_column!r} isn't a valid date/time column — "
            f"{unparseable_ratio:.0%} of values don't parse as a date",
            {"unparseable_ratio": round(unparseable_ratio, 3)},
        )
    timestamps = cast(pd.Series, parsed_df[time_column]).dropna().sort_values()
    if len(timestamps) < 3:
        return CheckResult("check_time_gaps", "info", "not enough timestamps to evaluate gaps", {})
    gaps = timestamps.diff().dropna()
    # Series.median() of datetime diffs is a Timedelta at runtime, despite the
    # stub typing it as a float.
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
