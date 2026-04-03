"""Differential privacy noise mechanisms for profiling."""

import math

import numpy as np


def laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    rng: np.random.Generator,
) -> float:
    """Add Laplace noise calibrated to sensitivity/epsilon.

    Args:
        value: True value to protect.
        sensitivity: L1 sensitivity of the query.
        epsilon: Privacy parameter (smaller = more private).
        rng: Seeded RNG.

    Returns:
        Noised value.
    """
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale)
    return value + noise


def gaussian_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
) -> float:
    """Add Gaussian noise calibrated for (epsilon, delta)-DP.

    Args:
        value: True value to protect.
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter.
        delta: Failure probability.
        rng: Seeded RNG.

    Returns:
        Noised value.
    """
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = rng.normal(0, sigma)
    return value + noise


def dp_histogram(
    counts: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Laplace noise to histogram bin counts.

    Args:
        counts: Array of bin counts.
        epsilon: Privacy parameter.
        rng: Seeded RNG.

    Returns:
        Noised counts (clipped to non-negative).
    """
    sensitivity = 1.0  # Adding/removing one person changes one bin by 1
    noise = rng.laplace(0, sensitivity / epsilon, size=len(counts))
    noised = counts.astype(np.float64) + noise
    return np.maximum(noised, 0.0)


def allocate_budget(
    n_queries: int,
    epsilon: float,
    delta: float,
) -> tuple[float, float]:
    """Split privacy budget across queries via sequential composition.

    Args:
        n_queries: Number of queries to support.
        epsilon: Total epsilon budget.
        delta: Total delta budget.

    Returns:
        Tuple of (per_query_epsilon, per_query_delta).
    """
    per_epsilon = epsilon / n_queries
    per_delta = delta / n_queries
    return per_epsilon, per_delta
