"""Categorical column generator."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import CategoricalStats, ColumnDefinition
from proof_of_replica.exceptions import GenerationError


def generate_categorical(
    col: ColumnDefinition,
    n: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Generate a categorical column via weighted random choice.

    Args:
        col: Column definition with CategoricalStats containing value_counts.
        n: Number of rows to generate.
        rng: Seeded numpy random Generator.

    Returns:
        A Polars Series of Utf8 dtype.

    Raises:
        GenerationError: If stats are missing or value_counts is empty.
    """
    if not isinstance(col.stats, CategoricalStats):
        msg = f"Column '{col.name}': categorical generator requires CategoricalStats"
        raise GenerationError(msg)

    if not col.stats.value_counts:
        msg = f"Column '{col.name}': value_counts is empty"
        raise GenerationError(msg)

    labels = list(col.stats.value_counts.keys())
    weights = np.array(list(col.stats.value_counts.values()), dtype=np.float64)

    # Normalize in case weights don't sum to 1
    total = weights.sum()
    if total > 0:
        weights = weights / total

    values = rng.choice(labels, size=n, p=weights)
    return pl.Series(name=col.name, values=values, dtype=pl.Utf8)
