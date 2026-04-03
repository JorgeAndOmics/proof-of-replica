"""Cross-column constraint enforcement for generated data."""

import logging

import polars as pl

from proof_of_replica.core.cross_constraints import (
    ArithmeticInvariant,
    CompositeUniqueness,
    ConditionalNull,
    ConstraintAction,
    CrossConstraintBlock,
    TemporalOrdering,
)
from proof_of_replica.exceptions import GenerationError

logger = logging.getLogger(__name__)


def apply_cross_constraints(
    df: pl.DataFrame,
    constraints: CrossConstraintBlock,
) -> pl.DataFrame:
    """Apply all cross-column constraints to a DataFrame.

    Args:
        df: Generated DataFrame.
        constraints: Cross-column constraint definitions.

    Returns:
        DataFrame with constraints enforced.
    """
    result = df

    for cn in constraints.conditional_nulls:
        result = _apply_conditional_null(result, cn)

    for to in constraints.temporal_ordering:
        result = _apply_temporal_ordering(result, to)

    for ai in constraints.arithmetic_invariants:
        result = _apply_arithmetic_invariant(result, ai)

    for cu in constraints.composite_uniqueness:
        result = _apply_composite_uniqueness(result, cu)

    return result


def _apply_conditional_null(
    df: pl.DataFrame, constraint: ConditionalNull
) -> pl.DataFrame:
    """Set column values to null/zero/value based on a condition."""
    if constraint.column not in df.columns:
        return df

    try:
        mask = df.select(pl.sql_expr(constraint.condition)).to_series()
    except Exception as exc:
        msg = f"Failed to evaluate condition '{constraint.condition}': {exc}"
        raise GenerationError(msg) from exc

    if constraint.action == ConstraintAction.SET_NULL:
        replacement = pl.lit(None)
    elif constraint.action == ConstraintAction.SET_ZERO:
        replacement = pl.lit(0)
    else:
        replacement = pl.lit(constraint.value)

    return df.with_columns(
        pl.when(mask)
        .then(replacement)
        .otherwise(pl.col(constraint.column))
        .alias(constraint.column)
    )


def _apply_temporal_ordering(
    df: pl.DataFrame, constraint: TemporalOrdering
) -> pl.DataFrame:
    """Swap date values where before > after."""
    if constraint.before not in df.columns or constraint.after not in df.columns:
        return df

    # Detect violations and swap
    return df.with_columns(
        pl.when(pl.col(constraint.before) > pl.col(constraint.after))
        .then(pl.col(constraint.after))
        .otherwise(pl.col(constraint.before))
        .alias(constraint.before),
        pl.when(pl.col(constraint.before) > pl.col(constraint.after))
        .then(pl.col(constraint.before))
        .otherwise(pl.col(constraint.after))
        .alias(constraint.after),
    )


def _apply_arithmetic_invariant(
    df: pl.DataFrame, constraint: ArithmeticInvariant
) -> pl.DataFrame:
    """Recompute a derived column from an expression."""
    try:
        computed = df.select(
            pl.sql_expr(constraint.expression).alias(constraint.derived)
        )
        return df.with_columns(computed)
    except Exception as exc:
        msg = f"Failed to evaluate expression '{constraint.expression}': {exc}"
        raise GenerationError(msg) from exc


def _apply_composite_uniqueness(
    df: pl.DataFrame, constraint: CompositeUniqueness
) -> pl.DataFrame:
    """Enforce uniqueness on a combination of columns.

    Drops duplicate rows based on the specified columns, keeping the first
    occurrence.
    """
    missing = [c for c in constraint.columns if c not in df.columns]
    if missing:
        return df

    return df.unique(subset=constraint.columns, keep="first")
