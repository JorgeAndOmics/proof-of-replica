"""Per-dtype noise perturbation for generated data."""

import datetime

import numpy as np
import polars as pl

from proof_of_replica.core.enums import ColumnDtype


def apply_noise(
    series: pl.Series,
    dtype: ColumnDtype,
    noise_level: float,
    rng: np.random.Generator,
) -> pl.Series:
    """Apply noise perturbation to a generated series.

    Args:
        series: The generated column data.
        dtype: Column data type (determines noise method).
        noise_level: Perturbation strength in [0.0, 1.0].
        rng: Seeded RNG.

    Returns:
        New series with noise applied. Null values are preserved.
    """
    if noise_level <= 0.0 or len(series) == 0:
        return series

    if dtype in (ColumnDtype.FLOAT64, ColumnDtype.INT64):
        return _noise_numeric(series, dtype, noise_level, rng)
    if dtype == ColumnDtype.CATEGORICAL:
        return _noise_categorical(series, noise_level, rng)
    if dtype == ColumnDtype.BOOLEAN:
        return _noise_boolean(series, noise_level, rng)
    if dtype == ColumnDtype.DATE:
        return _noise_date(series, noise_level, rng)

    # String and identifier columns: no noise applied
    return series


def _noise_numeric(
    series: pl.Series,
    dtype: ColumnDtype,
    noise_level: float,
    rng: np.random.Generator,
) -> pl.Series:
    """Additive Gaussian noise scaled to noise_level * column_std."""
    values = series.to_numpy().astype(np.float64)
    null_mask = series.is_null()

    std = np.nanstd(values) if np.nanstd(values) > 0 else 1.0
    noise = rng.normal(0, noise_level * std, size=len(values))
    noisy = values + noise

    if dtype == ColumnDtype.INT64:
        noisy = np.rint(noisy).astype(np.int64)
        result = pl.Series(name=series.name, values=noisy, dtype=pl.Int64)
    else:
        result = pl.Series(name=series.name, values=noisy, dtype=pl.Float64)

    if null_mask.any():
        result = result.set(null_mask, None)

    return result


def _noise_categorical(
    series: pl.Series,
    noise_level: float,
    rng: np.random.Generator,
) -> pl.Series:
    """Random label flipping with probability noise_level."""
    values = series.to_list()
    unique_labels = [v for v in series.unique().to_list() if v is not None]

    if len(unique_labels) < 2:
        return series

    flip_mask = rng.random(len(values)) < noise_level
    result = []
    for i, val in enumerate(values):
        if val is None or not flip_mask[i]:
            result.append(val)
        else:
            other = [lbl for lbl in unique_labels if lbl != val]
            result.append(rng.choice(other))

    return pl.Series(name=series.name, values=result, dtype=series.dtype)


def _noise_boolean(
    series: pl.Series,
    noise_level: float,
    rng: np.random.Generator,
) -> pl.Series:
    """Bit-flip with probability noise_level."""
    values = series.to_list()
    flip_mask = rng.random(len(values)) < noise_level

    result: list[bool | None] = []
    for i, val in enumerate(values):
        if val is None:
            result.append(None)
        elif flip_mask[i]:
            result.append(not val)
        else:
            result.append(val)

    return pl.Series(name=series.name, values=result, dtype=pl.Boolean)


def _noise_date(
    series: pl.Series,
    noise_level: float,
    rng: np.random.Generator,
) -> pl.Series:
    """Additive noise in days scaled to noise_level * date_range_days."""
    dates = series.to_list()
    non_null = [d for d in dates if d is not None]

    if not non_null:
        return series

    min_date = min(non_null)
    max_date = max(non_null)
    range_days = (max_date - min_date).days
    if range_days == 0:
        range_days = 1

    noise_days = rng.normal(0, noise_level * range_days, size=len(dates))

    result: list[datetime.date | None] = []
    for i, val in enumerate(dates):
        if val is None:
            result.append(None)
        else:
            delta = datetime.timedelta(days=round(noise_days[i]))
            result.append(val + delta)

    return pl.Series(name=series.name, values=result, dtype=pl.Date)
