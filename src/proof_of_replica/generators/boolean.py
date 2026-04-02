"""Boolean column generator."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import BooleanStats, ColumnDefinition
from proof_of_replica.exceptions import GenerationError


def generate_boolean(
    col: ColumnDefinition,
    n: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Generate a boolean column via Bernoulli draw.

    Args:
        col: Column definition with BooleanStats containing true_fraction.
        n: Number of rows to generate.
        rng: Seeded numpy random Generator.

    Returns:
        A Polars Series of Boolean dtype.

    Raises:
        GenerationError: If stats are missing or not BooleanStats.
    """
    if not isinstance(col.stats, BooleanStats):
        msg = f"Column '{col.name}': boolean generator requires BooleanStats"
        raise GenerationError(msg)

    values = rng.random(n) < col.stats.true_fraction
    return pl.Series(name=col.name, values=values, dtype=pl.Boolean)
