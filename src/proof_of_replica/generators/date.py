"""Date column generator."""

import datetime

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition, DateStats
from proof_of_replica.exceptions import GenerationError


def generate_date(
    col: ColumnDefinition,
    n: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Generate a date column with uniform distribution within a range.

    Args:
        col: Column definition with DateStats containing min/max dates.
        n: Number of rows.
        rng: Seeded RNG.

    Returns:
        Polars Series of Date dtype.

    Raises:
        GenerationError: If stats are missing or min/max dates are absent.
    """
    if not isinstance(col.stats, DateStats):
        msg = f"Column '{col.name}': date generator requires DateStats"
        raise GenerationError(msg)

    if col.stats.min is None or col.stats.max is None:
        msg = f"Column '{col.name}': date generator requires min and max dates"
        raise GenerationError(msg)

    date_min = datetime.date.fromisoformat(col.stats.min)
    date_max = datetime.date.fromisoformat(col.stats.max)

    ordinal_min = date_min.toordinal()
    ordinal_max = date_max.toordinal()

    ordinals = rng.integers(ordinal_min, ordinal_max + 1, size=n)
    dates = [datetime.date.fromordinal(int(o)) for o in ordinals]

    return pl.Series(name=col.name, values=dates, dtype=pl.Date)
