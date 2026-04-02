"""Generation pipeline orchestrator."""

import logging

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import (
    ColumnConstraints,
    ColumnDefinition,
)
from proof_of_replica.core.enums import ColumnDtype, ColumnRole
from proof_of_replica.core.schema import Profile
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.boolean import generate_boolean
from proof_of_replica.generators.categorical import generate_categorical
from proof_of_replica.generators.date import generate_date
from proof_of_replica.generators.identifier import generate_identifier
from proof_of_replica.generators.numeric import generate_numeric
from proof_of_replica.generators.string import generate_string
from proof_of_replica.utils.correlation import induce_correlation
from proof_of_replica.utils.missingness import inject_missingness
from proof_of_replica.utils.noise import apply_noise
from proof_of_replica.utils.privacy import check_privacy_distance

logger = logging.getLogger(__name__)

_DEFAULT_ROW_COUNT = 1000


def generate(
    profile: Profile,
    *,
    row_count: int | None = None,
    seed: int | None = None,
    original_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Generate a synthetic replica from a profile.

    Args:
        profile: Validated Profile instance.
        row_count: Override profile.row_count.
        seed: Override profile.seed.
        original_data: If provided, used for privacy distance check.

    Returns:
        A Polars DataFrame with all columns defined in the profile.

    Raises:
        GenerationError: If generation fails for any column.
    """
    n = _resolve_row_count(profile, row_count)
    rng = np.random.default_rng(seed if seed is not None else profile.seed)

    # Classify columns
    group_cols, other_cols = _classify_columns(profile)

    # Generate group columns first
    series_map: dict[str, pl.Series] = {}
    for col_def in group_cols:
        series_map[col_def.name] = _dispatch_generator(col_def, n, rng)

    # Generate remaining columns
    for col_def in other_cols:
        series_map[col_def.name] = _dispatch_generator(col_def, n, rng)

    # Assemble DataFrame in profile column order
    df = pl.DataFrame([series_map[col.name] for col in profile.columns])

    # Apply group effects
    for col_def in profile.columns:
        if col_def.group_effects is not None:
            df = _apply_group_effects(df, col_def)

    # Apply correlation induction
    if profile.correlations is not None:
        df = induce_correlation(
            df,
            profile.correlations.columns,
            profile.correlations.matrix,
            rng,
        )

    # Apply per-column constraints
    for col_def in profile.columns:
        if col_def.constraints is not None:
            df = df.with_columns(
                _apply_constraints(df[col_def.name], col_def.constraints).alias(
                    col_def.name
                )
            )

    # Inject missingness
    df = inject_missingness(df, profile, rng)

    # Apply noise
    noise_level = profile.defaults.noise_level
    for col_def in profile.columns:
        col_noise = noise_level
        if col_def.overrides is not None and col_def.overrides.noise_level is not None:
            col_noise = col_def.overrides.noise_level
        if col_noise > 0:
            df = df.with_columns(
                apply_noise(df[col_def.name], col_def.dtype, col_noise, rng).alias(
                    col_def.name
                )
            )

    # Privacy check
    if original_data is not None:
        stats = check_privacy_distance(df, original_data)
        logger.info("Privacy distance: %s", stats)

    return df


def _resolve_row_count(profile: Profile, override: int | None) -> int:
    """Determine the number of rows to generate."""
    if override is not None:
        return override
    if profile.row_count is not None:
        return profile.row_count
    return _DEFAULT_ROW_COUNT


def _classify_columns(
    profile: Profile,
) -> tuple[list[ColumnDefinition], list[ColumnDefinition]]:
    """Split columns into group columns and non-group columns."""
    group_cols = [c for c in profile.columns if c.role == ColumnRole.GROUP]
    other_cols = [c for c in profile.columns if c.role != ColumnRole.GROUP]
    return group_cols, other_cols


def _dispatch_generator(
    col: ColumnDefinition, n: int, rng: np.random.Generator
) -> pl.Series:
    """Route to the appropriate generator based on column dtype and config."""
    # Identifier with generator config
    if col.role == ColumnRole.IDENTIFIER and col.generator is not None:
        return generate_identifier(col, n, rng)

    # String with generator config
    if col.dtype == ColumnDtype.STRING and col.generator is not None:
        return generate_string(col, n, rng)

    match col.dtype:
        case ColumnDtype.FLOAT64 | ColumnDtype.INT64:
            return generate_numeric(col, n, rng)
        case ColumnDtype.CATEGORICAL:
            return generate_categorical(col, n, rng)
        case ColumnDtype.BOOLEAN:
            return generate_boolean(col, n, rng)
        case ColumnDtype.DATE:
            return generate_date(col, n, rng)
        case ColumnDtype.STRING:
            msg = f"Column '{col.name}': string column requires a generator config"
            raise GenerationError(msg)


def _apply_group_effects(df: pl.DataFrame, col_def: ColumnDefinition) -> pl.DataFrame:
    """Apply group-level shift and scale to a numeric column."""
    effects = col_def.group_effects
    if effects is None:
        return df

    group_col = effects.group_column
    if group_col not in df.columns:
        msg = f"Group column '{group_col}' not found in DataFrame"
        raise GenerationError(msg)

    # Build mapping DataFrame
    group_labels = list(effects.effects.keys())
    shifts = [effects.effects[g].shift for g in group_labels]
    scales = [effects.effects[g].scale_factor for g in group_labels]

    mapping = pl.DataFrame(
        {
            group_col: group_labels,
            "_shift": shifts,
            "_scale": scales,
        }
    )

    # Join and apply
    df = df.join(mapping, on=group_col, how="left")

    # Fill nulls (for group values not in effects) with identity transform
    df = df.with_columns(
        pl.col("_shift").fill_null(0.0),
        pl.col("_scale").fill_null(1.0),
    )

    df = df.with_columns(
        ((pl.col(col_def.name) + pl.col("_shift")) * pl.col("_scale")).alias(
            col_def.name
        )
    )

    return df.drop("_shift", "_scale")


def _apply_constraints(series: pl.Series, constraints: ColumnConstraints) -> pl.Series:
    """Clamp values and enforce integer constraint."""
    result = series
    if constraints.min is not None:
        result = result.clip(lower_bound=constraints.min)
    if constraints.max is not None:
        result = result.clip(upper_bound=constraints.max)
    if constraints.integer_valued and result.dtype in (pl.Float64, pl.Float32):
        result = result.round(0).cast(pl.Int64)

    return result
