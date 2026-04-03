"""Profiling engine — extract a Profile from a DataFrame."""

import hashlib
import logging
import warnings
from datetime import UTC, datetime

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from proof_of_replica.core.column_schema import (
    BooleanStats,
    CategoricalStats,
    ColumnDefinition,
    DateStats,
    DistributionConfig,
    GroupEffect,
    GroupEffects,
    NumericStats,
    StringStats,
)
from proof_of_replica.core.enums import ColumnDtype, ColumnRole
from proof_of_replica.core.schema import (
    CorrelationConfig,
    Profile,
    ProfilerConfig,
)
from proof_of_replica.engines._distribution_fit import fit_distribution
from proof_of_replica.engines._dp_noise import (
    allocate_budget,
    gaussian_mechanism,
    laplace_mechanism,
)
from proof_of_replica.engines._type_inference import infer_dtype

logger = logging.getLogger(__name__)


def profile_dataframe(
    df: pl.DataFrame,
    *,
    correlations: bool = False,
    group_column: str | None = None,
    seed: int = 42,
    config: ProfilerConfig | None = None,
    dp: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
) -> Profile:
    """Extract a statistical profile from a DataFrame.

    Args:
        df: Input data to profile.
        correlations: If True, compute and store pairwise correlation matrix.
        group_column: Column name for group-level statistics.
        seed: Random seed stored in the profile.
        config: Profiler thresholds. Uses defaults if None.
        dp: If True, apply differential privacy noise to all statistics.
        epsilon: DP epsilon parameter (total budget).
        delta: DP delta parameter.

    Returns:
        A validated Profile instance.
    """
    cfg = config or ProfilerConfig()
    n_rows = len(df)
    columns: list[ColumnDefinition] = []

    # DP setup
    dp_rng = np.random.default_rng(seed) if dp else None
    n_columns = len(df.columns)
    if dp:
        per_eps, per_delta = allocate_budget(max(n_columns, 1), epsilon, delta)
    else:
        per_eps, per_delta = epsilon, delta

    for col_name in df.columns:
        series = df[col_name]
        dtype = infer_dtype(series, n_rows, cfg)
        col_def = _profile_column(series, dtype, n_rows, cfg, group_column, df)
        if dp and dp_rng is not None:
            col_def = _apply_dp_to_column(col_def, per_eps, per_delta, dp_rng)
        columns.append(col_def)

    corr_config = None
    if correlations:
        corr_config = _compute_correlations(df, columns, cfg)

    source_hash = hashlib.sha256(
        f"{n_rows}:{','.join(df.columns)}".encode()
    ).hexdigest()

    privacy_block = None
    if dp:
        privacy_block = {
            "mechanism": "differential_privacy",
            "epsilon": epsilon,
            "delta": delta,
            "composition": "sequential",
        }

    return Profile(
        version="0.2.0",
        created_at=datetime.now(tz=UTC).isoformat(),
        source_hash=f"sha256:{source_hash}",
        seed=seed,
        row_count=n_rows,
        profiler=cfg,
        columns=columns,
        correlations=corr_config,
        privacy=privacy_block,
    )


def _apply_dp_to_column(
    col_def: ColumnDefinition,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
) -> ColumnDefinition:
    """Apply DP noise to a column's extracted statistics."""

    if col_def.stats is None:
        return col_def

    stats = col_def.stats
    noised: dict[str, object] = stats.model_dump()

    if isinstance(stats, NumericStats):
        sens = 1.0
        if stats.mean is not None:
            noised["mean"] = gaussian_mechanism(stats.mean, sens, epsilon, delta, rng)
        if stats.std is not None:
            noised["std"] = max(
                0.01, gaussian_mechanism(stats.std, sens, epsilon, delta, rng)
            )
        if stats.min is not None:
            noised["min"] = gaussian_mechanism(stats.min, sens, epsilon, delta, rng)
        if stats.max is not None:
            noised["max"] = gaussian_mechanism(stats.max, sens, epsilon, delta, rng)
        noised["null_fraction"] = max(
            0.0,
            min(1.0, laplace_mechanism(stats.null_fraction, 1.0 / 100, epsilon, rng)),
        )
    elif isinstance(stats, CategoricalStats):
        noised_vc = {}
        for label, frac in stats.value_counts.items():
            noised_vc[label] = max(
                0.0, laplace_mechanism(frac, 1.0 / 100, epsilon, rng)
            )
        total = sum(noised_vc.values())
        if total > 0:
            noised_vc = {k: v / total for k, v in noised_vc.items()}
        noised["value_counts"] = noised_vc
    elif isinstance(stats, BooleanStats):
        noised["true_fraction"] = max(
            0.0,
            min(1.0, laplace_mechanism(stats.true_fraction, 1.0 / 100, epsilon, rng)),
        )

    return ColumnDefinition.model_validate({**col_def.model_dump(), "stats": noised})


def _profile_column(
    series: pl.Series,
    dtype: ColumnDtype,
    n_rows: int,
    config: ProfilerConfig,
    group_column: str | None,
    df: pl.DataFrame,
) -> ColumnDefinition:
    """Build a ColumnDefinition from a single column."""
    role = _infer_role(series, dtype, n_rows, group_column)

    stats: (
        NumericStats | CategoricalStats | BooleanStats | DateStats | StringStats | None
    ) = None
    distribution: DistributionConfig | None = None
    group_effects: GroupEffects | None = None

    if dtype == ColumnDtype.BOOLEAN:
        stats = _extract_boolean_stats(series)
    elif dtype in (ColumnDtype.FLOAT64, ColumnDtype.INT64):
        stats, distribution = _extract_numeric_stats(series, config)
        if group_column and group_column in df.columns and group_column != series.name:
            group_effects = _extract_group_effects(series, df[group_column], stats)
    elif dtype == ColumnDtype.CATEGORICAL:
        stats = _extract_categorical_stats(series)
    elif dtype == ColumnDtype.DATE:
        stats = _extract_date_stats(series)
    else:
        stats = _extract_string_stats(series)

    return ColumnDefinition(
        name=series.name,
        dtype=dtype,
        role=role,
        stats=stats,
        distribution=distribution,
        group_effects=group_effects,
    )


def _infer_role(
    series: pl.Series,
    dtype: ColumnDtype,
    n_rows: int,
    group_column: str | None,
) -> ColumnRole:
    """Infer the semantic role of a column."""
    if series.name == group_column:
        return ColumnRole.GROUP
    if dtype == ColumnDtype.STRING and series.drop_nulls().n_unique() == n_rows:
        return ColumnRole.IDENTIFIER
    return ColumnRole.FEATURE


def _extract_boolean_stats(series: pl.Series) -> BooleanStats:
    """Extract statistics for a boolean column."""
    non_null = series.drop_nulls()
    if series.dtype != pl.Boolean:
        # Convert 0/1 or string to boolean for counting
        true_count = sum(
            1
            for v in non_null.to_list()
            if str(v).lower() in ("1", "true", "yes", "t", "y")
        )
        true_fraction = true_count / len(non_null) if len(non_null) > 0 else 0.0
    else:
        true_fraction = (
            float(non_null.sum()) / len(non_null) if len(non_null) > 0 else 0.0
        )

    return BooleanStats(
        true_fraction=true_fraction,
        null_fraction=series.null_count() / len(series) if len(series) > 0 else 0.0,
    )


def _extract_numeric_stats(
    series: pl.Series, config: ProfilerConfig
) -> tuple[NumericStats, DistributionConfig | None]:
    """Extract statistics and fit distribution for a numeric column."""
    numeric = series.cast(pl.Float64, strict=False).drop_nulls()
    values = numeric.to_numpy()

    if len(values) == 0:
        return NumericStats(null_fraction=1.0), None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skew_val = float(sp_stats.skew(values)) if len(values) > 2 else 0.0
        kurt_val = float(sp_stats.kurtosis(values)) if len(values) > 3 else 0.0

    stats = NumericStats(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        min=float(np.min(values)),
        max=float(np.max(values)),
        median=float(np.median(values)),
        skewness=skew_val if np.isfinite(skew_val) else 0.0,
        kurtosis=kurt_val if np.isfinite(kurt_val) else 0.0,
        null_fraction=series.null_count() / len(series) if len(series) > 0 else 0.0,
        percentiles={
            "5": float(np.percentile(values, 5)),
            "25": float(np.percentile(values, 25)),
            "75": float(np.percentile(values, 75)),
            "95": float(np.percentile(values, 95)),
        },
    )

    dist = fit_distribution(values, config) if len(values) >= 5 else None
    return stats, dist


def _extract_categorical_stats(series: pl.Series) -> CategoricalStats:
    """Extract statistics for a categorical column."""
    non_null = series.drop_nulls()
    total = len(non_null)
    value_counts: dict[str, float] = {}

    if total > 0:
        vc = non_null.value_counts()
        for row in vc.iter_rows(named=True):
            value_counts[str(row[series.name])] = row["count"] / total

    return CategoricalStats(
        cardinality=non_null.n_unique() if total > 0 else 0,
        null_fraction=series.null_count() / len(series) if len(series) > 0 else 0.0,
        value_counts=value_counts,
    )


def _extract_date_stats(series: pl.Series) -> DateStats:
    """Extract statistics for a date column."""
    if series.dtype == pl.Utf8:
        dates = series.str.to_date(format="%Y-%m-%d", strict=False).drop_nulls()
    else:
        dates = series.drop_nulls()

    if len(dates) == 0:
        return DateStats(null_fraction=series.null_count() / max(len(series), 1))

    return DateStats(
        min=str(dates.min()),
        max=str(dates.max()),
        null_fraction=series.null_count() / len(series) if len(series) > 0 else 0.0,
    )


def _extract_string_stats(series: pl.Series) -> StringStats:
    """Extract statistics for a string column."""
    non_null = series.drop_nulls().cast(pl.Utf8, strict=False)

    if len(non_null) == 0:
        return StringStats(
            null_fraction=series.null_count() / max(len(series), 1),
        )

    lengths = non_null.str.len_bytes()
    return StringStats(
        null_fraction=series.null_count() / len(series) if len(series) > 0 else 0.0,
        mean_length=float(lengths.mean()),  # type: ignore[arg-type]
        max_length=int(lengths.max()),  # type: ignore[arg-type]
    )


def _compute_correlations(
    df: pl.DataFrame,
    columns: list[ColumnDefinition],
    config: ProfilerConfig,
) -> CorrelationConfig | None:
    """Compute pairwise Pearson correlations for numeric columns."""
    numeric_cols = [
        c.name for c in columns if c.dtype in (ColumnDtype.FLOAT64, ColumnDtype.INT64)
    ]

    if len(numeric_cols) < 2:
        return None

    numeric_df = df.select(numeric_cols).cast(dict.fromkeys(numeric_cols, pl.Float64))
    arr = numeric_df.to_numpy()

    # Drop rows with any NaN
    mask = ~np.any(np.isnan(arr), axis=1)
    clean = arr[mask]

    if len(clean) < 3:
        return None

    corr_matrix = np.corrcoef(clean, rowvar=False)

    # Filter by threshold — zero out small correlations
    filtered = corr_matrix.copy()
    np.fill_diagonal(filtered, 1.0)
    below_threshold = np.abs(filtered) < config.correlation_threshold
    np.fill_diagonal(below_threshold, False)
    filtered[below_threshold] = 0.0

    return CorrelationConfig(
        columns=numeric_cols,
        matrix=filtered.tolist(),
    )


def _extract_group_effects(
    series: pl.Series,
    group_series: pl.Series,
    global_stats: NumericStats,
) -> GroupEffects | None:
    """Compute group-level shift and scale effects for a numeric column."""
    if global_stats.mean is None or global_stats.std is None:
        return None
    if global_stats.std == 0:
        return None

    global_mean = global_stats.mean
    global_std = global_stats.std

    combined = pl.DataFrame(
        {
            series.name: series.cast(pl.Float64, strict=False),
            group_series.name: group_series,
        }
    ).drop_nulls()

    groups = combined.group_by(group_series.name).agg(
        pl.col(series.name).mean().alias("group_mean"),
        pl.col(series.name).std().alias("group_std"),
    )

    effects: dict[str, GroupEffect] = {}
    for row in groups.iter_rows(named=True):
        label = str(row[group_series.name])
        group_mean = row["group_mean"] or global_mean
        group_std = row["group_std"] or global_std
        shift = group_mean - global_mean
        scale = group_std / global_std if global_std > 0 else 1.0
        effects[label] = GroupEffect(shift=shift, scale_factor=scale)

    return GroupEffects(group_column=group_series.name, effects=effects)
