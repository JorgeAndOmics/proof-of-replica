"""Correlation induction via Gaussian copula (Cholesky decomposition)."""

import logging

import numpy as np
import polars as pl
import scipy.stats

from proof_of_replica.exceptions import GenerationError

logger = logging.getLogger(__name__)


def induce_correlation(
    df: pl.DataFrame,
    columns: list[str],
    target_matrix: list[list[float]],
    _rng: np.random.Generator,
) -> pl.DataFrame:
    """Induce a target correlation structure on numeric columns.

    Uses a Gaussian copula approach:
    1. Transform each column to uniform marginals via rank-based CDF
    2. Transform to standard normal
    3. Apply Cholesky factor of target correlation matrix
    4. Transform back to uniform, then to original marginals

    Args:
        df: DataFrame with the columns to correlate.
        columns: Column names to correlate.
        target_matrix: Target correlation matrix (NxN).
        rng: Seeded RNG (unused in current impl, reserved for future).

    Returns:
        DataFrame with correlated columns replacing the originals.

    Raises:
        GenerationError: If the target matrix is not positive semi-definite.
    """
    if len(columns) < 2:
        return df

    n = len(df)
    target = np.array(target_matrix, dtype=np.float64)

    # Ensure positive semi-definite via eigenvalue clipping
    target = _nearest_psd(target)

    try:
        chol = np.linalg.cholesky(target)
    except np.linalg.LinAlgError as exc:
        msg = "Target correlation matrix is not positive semi-definite"
        raise GenerationError(msg) from exc

    # Extract original columns as numpy arrays and store their sorted values
    originals: dict[str, np.ndarray] = {}
    for col_name in columns:
        originals[col_name] = df[col_name].to_numpy().astype(np.float64)

    # Step 1: Transform each column to uniform marginals via empirical CDF (ranks)
    uniform = np.zeros((n, len(columns)))
    for j, col_name in enumerate(columns):
        vals = originals[col_name]
        ranks = scipy.stats.rankdata(vals, method="ordinal")
        # Map to (0, 1) avoiding exact 0 and 1 for ppf
        uniform[:, j] = ranks / (n + 1)

    # Step 2: Transform to standard normal
    normal = scipy.stats.norm.ppf(uniform)

    # Step 3: Apply Cholesky factor to induce correlations
    correlated_normal = (chol @ normal.T).T

    # Step 4: Transform back to uniform
    correlated_uniform = scipy.stats.norm.cdf(correlated_normal)

    # Step 5: Transform back to original marginals via quantile mapping
    result = df.clone()
    for j, col_name in enumerate(columns):
        sorted_original = np.sort(originals[col_name])
        # Map uniform values to indices in the sorted original
        indices = np.clip(
            (correlated_uniform[:, j] * n).astype(np.int64),
            0,
            n - 1,
        )
        new_values = sorted_original[indices]
        result = result.with_columns(
            pl.Series(name=col_name, values=new_values, dtype=pl.Float64)
        )

    return result


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest positive semi-definite matrix.

    Clips negative eigenvalues to a small positive value.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    result: np.ndarray = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return result
