"""Matplotlib plotting helpers for fidelity reports.

Each function returns a base64-encoded PNG string for HTML embedding.
"""

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")


def plot_numeric_distribution(
    series: pl.Series,
    *,
    original: pl.Series | None = None,
    title: str = "",
) -> str:
    """Plot histogram + KDE for a numeric column.

    Args:
        series: Replica column data.
        original: Optional original column for side-by-side comparison.
        title: Plot title.

    Returns:
        Base64-encoded PNG image string.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    vals = series.drop_nulls().cast(pl.Float64).to_numpy()
    if len(vals) > 0:
        ax.hist(
            vals, bins=30, alpha=0.6, density=True, label="Replica", color="#4c72b0"
        )

    if original is not None:
        orig_vals = original.drop_nulls().cast(pl.Float64).to_numpy()
        if len(orig_vals) > 0:
            ax.hist(
                orig_vals,
                bins=30,
                alpha=0.4,
                density=True,
                label="Original",
                color="#dd8452",
            )

    ax.set_title(title or series.name)
    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

    return _fig_to_base64(fig)


def plot_categorical_distribution(
    series: pl.Series,
    *,
    original: pl.Series | None = None,
    title: str = "",
) -> str:
    """Plot bar chart for a categorical column.

    Args:
        series: Replica column data.
        original: Optional original column for comparison.
        title: Plot title.

    Returns:
        Base64-encoded PNG string.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    repl_vc = series.drop_nulls().value_counts().sort(series.name)
    labels = [str(v) for v in repl_vc[series.name].to_list()]
    repl_counts = repl_vc["count"].to_numpy().astype(float)
    repl_fracs = (
        repl_counts / repl_counts.sum() if repl_counts.sum() > 0 else repl_counts
    )

    x = np.arange(len(labels))
    width = 0.35

    if original is not None:
        orig_vc = original.drop_nulls().value_counts().sort(original.name)
        orig_counts = np.zeros(len(labels))
        for i, lbl in enumerate(labels):
            row = orig_vc.filter(pl.col(original.name) == lbl)
            if len(row) > 0:
                orig_counts[i] = row["count"][0]
        orig_fracs = (
            orig_counts / orig_counts.sum() if orig_counts.sum() > 0 else orig_counts
        )
        ax.bar(
            x - width / 2,
            orig_fracs,
            width,
            label="Original",
            color="#dd8452",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            repl_fracs,
            width,
            label="Replica",
            color="#4c72b0",
            alpha=0.7,
        )
    else:
        ax.bar(x, repl_fracs, width, color="#4c72b0", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title or series.name)
    ax.set_ylabel("Fraction")
    if original is not None:
        ax.legend()

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_correlation_heatmap(
    matrix: np.ndarray,
    columns: list[str],
    *,
    title: str = "Correlation Matrix",
) -> str:
    """Plot a correlation matrix heatmap.

    Args:
        matrix: 2D correlation matrix.
        columns: Column labels.
        title: Plot title.

    Returns:
        Base64-encoded PNG string.
    """
    fig, ax = plt.subplots(
        figsize=(max(4, len(columns) * 0.8), max(4, len(columns) * 0.8))
    )
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return _fig_to_base64(fig)


def _fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
