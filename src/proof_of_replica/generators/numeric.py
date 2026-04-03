"""Numeric column generator (float64 and int64)."""

import json
from collections.abc import Callable

import numpy as np
import polars as pl
import scipy.stats

from proof_of_replica.core.column_schema import ColumnDefinition, NumericStats
from proof_of_replica.core.enums import ColumnDtype, DistributionFamily
from proof_of_replica.exceptions import GenerationError

# Mapping from DistributionFamily to scipy.stats frozen distribution constructor.
# Each entry is a callable that takes (params, rng) and returns samples.
_FAMILY_TO_SCIPY: dict[DistributionFamily, Callable[..., scipy.stats.rv_continuous]] = {
    DistributionFamily.NORMAL: scipy.stats.norm,
    DistributionFamily.LOGNORMAL: scipy.stats.lognorm,
    DistributionFamily.UNIFORM: scipy.stats.uniform,
    DistributionFamily.BETA: scipy.stats.beta,
    DistributionFamily.GAMMA: scipy.stats.gamma,
    DistributionFamily.EXPONENTIAL: scipy.stats.expon,
}


def _sample_parametric(
    family: DistributionFamily,
    params: dict[str, float | str],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a parametric distribution using scipy.stats.

    Args:
        family: Distribution family.
        params: Distribution parameters (family-specific).
        n: Number of samples.
        rng: Seeded RNG.

    Returns:
        Array of n samples.

    Raises:
        GenerationError: If family is unsupported or params are invalid.
    """
    float_params = {k: float(v) for k, v in params.items()}

    dist_cls = _FAMILY_TO_SCIPY.get(family)
    if dist_cls is None:  # pragma: no cover — empirical families handled separately
        msg = f"Unsupported distribution family: {family}"
        raise GenerationError(msg)

    dist = dist_cls(**float_params)
    samples: np.ndarray = dist.rvs(size=n, random_state=rng)
    return samples


def _sample_empirical_kde(
    params: dict[str, float | str],
    n: int,
    rng: np.random.Generator,
    stats: NumericStats,
) -> np.ndarray:
    """Sample from an empirical KDE distribution.

    Uses a normal distribution centered at the mean with the stored bandwidth
    as a scale approximation. This is a simplified KDE reconstruction for v1.
    """
    mean = stats.mean if stats.mean is not None else 0.0
    bandwidth = float(params.get("bandwidth", 1.0))
    return rng.normal(loc=mean, scale=bandwidth, size=n)


def _sample_empirical_histogram(
    params: dict[str, float | str],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from an empirical histogram via inverse CDF.

    Params should contain 'edges' and 'counts' as string-encoded lists,
    or the caller should pre-parse them. For v1, we fall back to uniform
    sampling within each bin, weighted by bin counts.
    """
    edges_raw = params.get("edges")
    counts_raw = params.get("counts")

    if edges_raw is None or counts_raw is None:
        msg = "empirical_histogram requires 'edges' and 'counts' in params"
        raise GenerationError(msg)

    # Params are always strings (Pydantic enforces float|str, lists come as JSON strings)
    if isinstance(edges_raw, str):
        edges = np.array(json.loads(edges_raw), dtype=np.float64)
    else:  # pragma: no cover — Pydantic enforces str for list-like params
        edges = np.array(edges_raw, dtype=np.float64)

    if isinstance(counts_raw, str):
        counts = np.array(json.loads(counts_raw), dtype=np.float64)
    else:  # pragma: no cover
        counts = np.array(counts_raw, dtype=np.float64)

    # Normalize counts to probabilities
    probs = counts / counts.sum()

    # Sample bin indices weighted by counts
    bin_indices = rng.choice(len(probs), size=n, p=probs)

    # Uniform sample within each bin
    lows = edges[bin_indices]
    highs = edges[bin_indices + 1]
    return rng.uniform(lows, highs)


def _sample_fallback(
    stats: NumericStats,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fall back to normal(mean, std) when no distribution is specified.

    If std is missing, estimates as max(abs(mean) * 0.1, 1.0).
    """
    if stats.mean is None:
        msg = "Cannot generate numeric column without mean in stats"
        raise GenerationError(msg)

    mean = stats.mean
    std = stats.std if stats.std is not None else max(abs(mean) * 0.1, 1.0)
    return rng.normal(loc=mean, scale=std, size=n)


def generate_numeric(
    col: ColumnDefinition,
    n: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Generate a numeric column (float64 or int64).

    Args:
        col: Column definition with optional distribution and stats.
        n: Number of rows.
        rng: Seeded RNG.

    Returns:
        Polars Series with Float64 or Int64 dtype.

    Raises:
        GenerationError: If required stats/distribution info is missing.
    """
    if not isinstance(col.stats, NumericStats):
        msg = f"Column '{col.name}': numeric generator requires NumericStats"
        raise GenerationError(msg)

    if col.distribution is not None:
        family = col.distribution.family
        params = col.distribution.params

        if family == DistributionFamily.EMPIRICAL_KDE:
            values = _sample_empirical_kde(params, n, rng, col.stats)
        elif family == DistributionFamily.EMPIRICAL_HISTOGRAM:
            values = _sample_empirical_histogram(params, n, rng)
        else:
            values = _sample_parametric(family, params, n, rng)
    else:
        values = _sample_fallback(col.stats, n, rng)

    if col.dtype == ColumnDtype.INT64:
        values = np.rint(values).astype(np.int64)
        return pl.Series(name=col.name, values=values, dtype=pl.Int64)

    return pl.Series(name=col.name, values=values, dtype=pl.Float64)
