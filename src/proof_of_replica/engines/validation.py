"""Validation engine — check replica fidelity against a profile."""

import logging
from enum import StrEnum

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from scipy import stats as sp_stats

from proof_of_replica.core.column_schema import (
    BooleanStats,
    CategoricalStats,
    ColumnDefinition,
    NumericStats,
)
from proof_of_replica.core.enums import ColumnDtype
from proof_of_replica.core.schema import Profile, ValidationConfig
from proof_of_replica.generators.numeric import _FAMILY_TO_SCIPY

logger = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    """Status of a single validation check."""

    PASS = "pass"  # noqa: S105
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    """Result of a single validation check."""

    model_config = ConfigDict(frozen=True)

    check_name: str
    column: str | None = None
    status: CheckStatus
    message: str
    detail: dict[str, float | str] | None = None


class ValidationResult(BaseModel):
    """Aggregate result of all validation checks."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    checks: list[CheckResult]
    summary: dict[str, int]


def validate(df: pl.DataFrame, profile: Profile) -> ValidationResult:
    """Validate a replica DataFrame against a profile.

    Args:
        df: The synthetic replica.
        profile: The profile it was generated from.

    Returns:
        ValidationResult with per-check results and summary.
    """
    checks: list[CheckResult] = []

    checks.extend(_check_columns(df, profile))

    for col_def in profile.columns:
        if col_def.name not in df.columns:
            continue
        series = df[col_def.name]
        thresholds = _resolve_thresholds(col_def, profile.validation)

        checks.extend(_check_column(series, col_def, thresholds))

    if profile.correlations is not None:
        checks.extend(_check_correlations(df, profile))

    summary = {s.value: 0 for s in CheckStatus}
    for c in checks:
        summary[c.status.value] += 1

    return ValidationResult(
        passed=summary.get("fail", 0) == 0,
        checks=checks,
        summary=summary,
    )


def _resolve_thresholds(
    col_def: ColumnDefinition, global_config: ValidationConfig
) -> ValidationConfig:
    """Merge per-column overrides onto global thresholds."""
    if col_def.validation_overrides is None:
        return global_config

    overrides = col_def.validation_overrides
    return ValidationConfig(
        ks_pvalue=overrides.ks_pvalue or global_config.ks_pvalue,
        chisq_pvalue=overrides.chisq_pvalue or global_config.chisq_pvalue,
        null_tolerance=overrides.null_tolerance
        if overrides.null_tolerance is not None
        else global_config.null_tolerance,
        range_tolerance=overrides.range_tolerance
        if overrides.range_tolerance is not None
        else global_config.range_tolerance,
        boolean_tolerance=overrides.boolean_tolerance
        if overrides.boolean_tolerance is not None
        else global_config.boolean_tolerance,
        correlation_frobenius=overrides.correlation_frobenius
        if overrides.correlation_frobenius is not None
        else global_config.correlation_frobenius,
    )


def _check_columns(df: pl.DataFrame, profile: Profile) -> list[CheckResult]:
    """Check that column names match."""
    expected = [c.name for c in profile.columns]
    actual = df.columns

    if actual == expected:
        return [
            CheckResult(
                check_name="column_names",
                status=CheckStatus.PASS,
                message="Column names match",
            )
        ]

    return [
        CheckResult(
            check_name="column_names",
            status=CheckStatus.FAIL,
            message=f"Column mismatch: expected {expected}, got {actual}",
        )
    ]


def _check_column(
    series: pl.Series,
    col_def: ColumnDefinition,
    thresholds: ValidationConfig,
) -> list[CheckResult]:
    """Run all applicable checks for a single column."""
    results: list[CheckResult] = []

    # Null fraction check
    if col_def.stats is not None and hasattr(col_def.stats, "null_fraction"):
        expected_null = getattr(col_def.stats, "null_fraction", 0.0)
        actual_null = series.null_count() / max(len(series), 1)
        diff = abs(actual_null - expected_null)

        results.append(
            CheckResult(
                check_name="null_fraction",
                column=col_def.name,
                status=CheckStatus.PASS
                if diff <= thresholds.null_tolerance
                else CheckStatus.FAIL,
                message=f"Null fraction: expected {expected_null:.3f}, got {actual_null:.3f}",
                detail={"expected": expected_null, "actual": actual_null, "diff": diff},
            )
        )

    # Type-specific checks
    if col_def.dtype in (ColumnDtype.FLOAT64, ColumnDtype.INT64):
        results.extend(_check_numeric(series, col_def, thresholds))
    elif col_def.dtype == ColumnDtype.CATEGORICAL:
        results.extend(_check_categorical(series, col_def, thresholds))
    elif col_def.dtype == ColumnDtype.BOOLEAN:
        results.extend(_check_boolean(series, col_def, thresholds))

    return results


def _check_numeric(
    series: pl.Series,
    col_def: ColumnDefinition,
    thresholds: ValidationConfig,
) -> list[CheckResult]:
    """Check value range and distribution for numeric columns."""
    results: list[CheckResult] = []
    if not isinstance(col_def.stats, NumericStats):
        return results

    stats = col_def.stats
    values = series.drop_nulls().cast(pl.Float64).to_numpy()

    # Range check
    if stats.min is not None and stats.max is not None and len(values) > 0:
        profile_range = stats.max - stats.min
        if profile_range > 0:
            min_dev = abs(float(np.min(values)) - stats.min) / profile_range
            max_dev = abs(float(np.max(values)) - stats.max) / profile_range
            worst = max(min_dev, max_dev)

            results.append(
                CheckResult(
                    check_name="value_range",
                    column=col_def.name,
                    status=CheckStatus.PASS
                    if worst <= thresholds.range_tolerance
                    else CheckStatus.FAIL,
                    message=f"Range deviation: {worst:.3f} (tolerance: {thresholds.range_tolerance})",
                    detail={"min_dev": min_dev, "max_dev": max_dev},
                )
            )

    # KS test against distribution
    if col_def.distribution is not None and len(values) > 10:
        results.extend(_check_ks_test(values, col_def, thresholds))

    return results


def _check_ks_test(
    values: np.ndarray,
    col_def: ColumnDefinition,
    thresholds: ValidationConfig,
) -> list[CheckResult]:
    """Run KS test against the profile's distribution."""
    if col_def.distribution is None:
        return []

    dist_cls = _FAMILY_TO_SCIPY.get(col_def.distribution.family)
    if dist_cls is None:
        return []

    float_params = {k: float(v) for k, v in col_def.distribution.params.items()}
    try:
        dist = dist_cls(**float_params)
        _, p_value = sp_stats.kstest(values, dist.cdf)
    except Exception:  # pragma: no cover
        return []

    return [
        CheckResult(
            check_name="ks_test",
            column=col_def.name,
            status=CheckStatus.PASS
            if p_value >= thresholds.ks_pvalue
            else CheckStatus.WARN,
            message=f"KS test p-value: {p_value:.4f} (threshold: {thresholds.ks_pvalue})",
            detail={"p_value": p_value},
        )
    ]


def _check_categorical(
    series: pl.Series,
    col_def: ColumnDefinition,
    thresholds: ValidationConfig,
) -> list[CheckResult]:
    """Check cardinality and chi-squared for categorical columns."""
    results: list[CheckResult] = []
    if not isinstance(col_def.stats, CategoricalStats):
        return results

    non_null = series.drop_nulls()

    # Cardinality check
    if col_def.stats.cardinality is not None:
        actual_card = non_null.n_unique()
        results.append(
            CheckResult(
                check_name="cardinality",
                column=col_def.name,
                status=CheckStatus.PASS
                if actual_card == col_def.stats.cardinality
                else CheckStatus.FAIL,
                message=f"Cardinality: expected {col_def.stats.cardinality}, got {actual_card}",
            )
        )

    # Chi-squared test
    if col_def.stats.value_counts and len(non_null) > 10:
        expected_labels = list(col_def.stats.value_counts.keys())
        expected_probs = np.array(list(col_def.stats.value_counts.values()))
        expected_probs = expected_probs / expected_probs.sum()

        vc = non_null.value_counts()
        actual_counts = np.zeros(len(expected_labels))
        for i, label in enumerate(expected_labels):
            row = vc.filter(pl.col(col_def.name) == label)
            if len(row) > 0:
                actual_counts[i] = row["count"][0]

        expected_counts = expected_probs * actual_counts.sum()
        valid = expected_counts > 0
        if valid.sum() > 1:
            _, p_value = sp_stats.chisquare(
                actual_counts[valid], expected_counts[valid]
            )
            results.append(
                CheckResult(
                    check_name="chisq_test",
                    column=col_def.name,
                    status=CheckStatus.PASS
                    if p_value >= thresholds.chisq_pvalue
                    else CheckStatus.WARN,
                    message=f"Chi-squared p-value: {p_value:.4f}",
                    detail={"p_value": float(p_value)},
                )
            )

    return results


def _check_boolean(
    series: pl.Series,
    col_def: ColumnDefinition,
    thresholds: ValidationConfig,
) -> list[CheckResult]:
    """Check boolean true fraction."""
    if not isinstance(col_def.stats, BooleanStats):
        return []

    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return []

    actual_frac = float(non_null.sum()) / len(non_null)
    diff = abs(actual_frac - col_def.stats.true_fraction)

    return [
        CheckResult(
            check_name="boolean_fraction",
            column=col_def.name,
            status=CheckStatus.PASS
            if diff <= thresholds.boolean_tolerance
            else CheckStatus.FAIL,
            message=f"True fraction: expected {col_def.stats.true_fraction:.3f}, got {actual_frac:.3f}",
            detail={"expected": col_def.stats.true_fraction, "actual": actual_frac},
        )
    ]


def _check_correlations(df: pl.DataFrame, profile: Profile) -> list[CheckResult]:
    """Check correlation matrix similarity via Frobenius norm."""
    if profile.correlations is None:
        return []

    cols = profile.correlations.columns
    target = np.array(profile.correlations.matrix)
    threshold = profile.validation.correlation_frobenius

    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return []

    numeric_df = df.select(available).cast(dict.fromkeys(available, pl.Float64))
    arr = numeric_df.drop_nulls().to_numpy()

    if len(arr) < 3:
        return []

    actual = np.corrcoef(arr, rowvar=False)

    # Subset target to available columns
    idx = [cols.index(c) for c in available]
    target_sub = target[np.ix_(idx, idx)]

    frob = float(np.linalg.norm(actual - target_sub, "fro"))

    return [
        CheckResult(
            check_name="correlation_frobenius",
            status=CheckStatus.PASS if frob <= threshold else CheckStatus.FAIL,
            message=f"Correlation Frobenius norm: {frob:.4f} (threshold: {threshold})",
            detail={"frobenius_norm": frob},
        )
    ]
