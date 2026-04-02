"""Missingness injection for generated data."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import Profile


def inject_missingness(
    df: pl.DataFrame,
    profile: Profile,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Inject null values into a DataFrame based on profile stats.

    Applies MCAR (Missing Completely At Random) nulls per column based
    on each column's null_fraction, then applies co-missingness patterns.

    Args:
        df: Generated DataFrame.
        profile: Profile with column stats and optional missingness config.
        rng: Seeded RNG.

    Returns:
        DataFrame with nulls injected.
    """
    result = df.clone()

    # Per-column MCAR nulls
    for col_def in profile.columns:
        null_fraction = _get_null_fraction(col_def)
        if null_fraction <= 0.0:
            continue
        if col_def.name not in result.columns:
            continue

        mask = rng.random(len(result)) < null_fraction
        null_indices = np.where(mask)[0].tolist()
        if null_indices:
            result = result.with_columns(
                result[col_def.name].scatter(null_indices, None).alias(col_def.name)
            )

    # Co-missingness
    if profile.missingness is not None and profile.missingness.co_missing is not None:
        result = _apply_co_missingness(result, profile, rng)

    return result


def _get_null_fraction(col_def: ColumnDefinition) -> float:
    """Extract null_fraction from column stats, defaulting to 0."""
    if col_def.stats is None:
        return 0.0
    if hasattr(col_def.stats, "null_fraction"):
        return getattr(col_def.stats, "null_fraction", 0.0)
    return 0.0


def _apply_co_missingness(
    df: pl.DataFrame,
    profile: Profile,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Apply co-missingness patterns from the profile.

    For each pair "col_a,col_b" with probability p: when col_a is null,
    set col_b to null with probability p.
    """
    if profile.missingness is None or profile.missingness.co_missing is None:
        return df

    result = df.clone()

    for pair_key, prob in profile.missingness.co_missing.items():
        parts = pair_key.split(",")
        if len(parts) != 2:
            continue

        col_a, col_b = parts[0].strip(), parts[1].strip()
        if col_a not in result.columns or col_b not in result.columns:
            continue

        a_null = result[col_a].is_null()
        co_mask = a_null & (pl.Series(rng.random(len(result))) < prob)
        null_indices = np.where(co_mask.to_numpy())[0].tolist()
        if null_indices:
            result = result.with_columns(
                result[col_b].scatter(null_indices, None).alias(col_b)
            )

    return result
