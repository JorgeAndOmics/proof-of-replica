"""Privacy distance check between synthetic and original data."""

import logging

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def check_privacy_distance(
    synthetic: pl.DataFrame,
    original: pl.DataFrame,
) -> dict[str, float]:
    """Compute nearest-neighbor distances between synthetic and original data.

    Uses numeric columns only. Reports minimum, mean, and 5th percentile
    distances. Warns if any synthetic row is identical to an original row.

    Args:
        synthetic: Generated replica DataFrame.
        original: Real data DataFrame.

    Returns:
        Dict with 'min', 'mean', 'p5' distance statistics.
    """
    # Find shared numeric columns
    numeric_cols = [
        c
        for c in synthetic.columns
        if c in original.columns
        and synthetic[c].dtype in (pl.Float64, pl.Int64)
        and original[c].dtype in (pl.Float64, pl.Int64)
    ]

    if not numeric_cols:
        logger.warning("No shared numeric columns for privacy check")
        return {"min": float("inf"), "mean": float("inf"), "p5": float("inf")}

    # Extract and normalize numeric data
    syn_arr = synthetic.select(numeric_cols).to_numpy().astype(np.float64)
    orig_arr = original.select(numeric_cols).to_numpy().astype(np.float64)

    # Handle NaN by replacing with column means
    for j in range(syn_arr.shape[1]):
        col_mean = np.nanmean(np.concatenate([syn_arr[:, j], orig_arr[:, j]]))
        syn_arr[np.isnan(syn_arr[:, j]), j] = col_mean
        orig_arr[np.isnan(orig_arr[:, j]), j] = col_mean

    # Normalize columns to [0, 1]
    combined = np.vstack([syn_arr, orig_arr])
    col_min = combined.min(axis=0)
    col_range = combined.max(axis=0) - col_min
    col_range[col_range == 0] = 1.0
    syn_norm = (syn_arr - col_min) / col_range
    orig_norm = (orig_arr - col_min) / col_range

    # Compute nearest-neighbor distances
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(orig_norm)
    distances, _ = nn.kneighbors(syn_norm)
    dists = distances.flatten()

    min_dist = float(dists.min())
    mean_dist = float(dists.mean())
    p5_dist = float(np.percentile(dists, 5))

    if min_dist == 0.0:
        logger.warning(
            "Privacy warning: at least one synthetic row is identical "
            "to an original row (distance=0)"
        )

    return {"min": min_dist, "mean": mean_dist, "p5": p5_dist}
