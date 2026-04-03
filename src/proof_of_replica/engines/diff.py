"""Diff engine — compare original data against a synthetic replica."""

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from scipy import stats as sp_stats

from proof_of_replica.utils.privacy import check_privacy_distance


class StatComparison(BaseModel):
    """Side-by-side comparison of a single statistic."""

    model_config = ConfigDict(frozen=True)

    original: float | str | None = None
    replica: float | str | None = None
    diff: float | None = None


class TestResult(BaseModel):
    """Result of a statistical test."""

    model_config = ConfigDict(frozen=True)

    test_name: str
    statistic: float
    p_value: float


class ColumnDiff(BaseModel):
    """Comparison results for a single column."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    stats_comparison: dict[str, StatComparison]
    distribution_test: TestResult | None = None


class DiffResult(BaseModel):
    """Complete comparison between original and replica datasets."""

    model_config = ConfigDict(frozen=True)

    column_diffs: list[ColumnDiff]
    privacy: dict[str, float] | None = None


def diff_datasets(
    original: pl.DataFrame,
    replica: pl.DataFrame,
    *,
    privacy: bool = False,
) -> DiffResult:
    """Compare an original dataset against its synthetic replica.

    Args:
        original: The real dataset.
        replica: The synthetic replica.
        privacy: If True, compute nearest-neighbor privacy distances.

    Returns:
        DiffResult with per-column comparisons and optional privacy metrics.
    """
    column_diffs: list[ColumnDiff] = []

    shared_cols = [c for c in original.columns if c in replica.columns]

    for col_name in shared_cols:
        orig_series = original[col_name]
        repl_series = replica[col_name]
        column_diffs.append(_diff_column(orig_series, repl_series))

    privacy_stats = None
    if privacy:
        privacy_stats = check_privacy_distance(replica, original)

    return DiffResult(column_diffs=column_diffs, privacy=privacy_stats)


def _diff_column(orig: pl.Series, repl: pl.Series) -> ColumnDiff:
    """Compute diff for a single column."""
    dtype_str = str(orig.dtype)

    if orig.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
        return _diff_numeric(orig, repl, dtype_str)
    if orig.dtype == pl.Boolean:
        return _diff_boolean(orig, repl, dtype_str)
    if orig.dtype == pl.Date:
        return _diff_date(orig, repl, dtype_str)

    return _diff_categorical(orig, repl, dtype_str)


def _diff_numeric(orig: pl.Series, repl: pl.Series, dtype_str: str) -> ColumnDiff:
    """Compare numeric columns."""
    o = orig.drop_nulls().cast(pl.Float64).to_numpy()
    r = repl.drop_nulls().cast(pl.Float64).to_numpy()

    stats: dict[str, StatComparison] = {}

    if len(o) > 0 and len(r) > 0:
        pairs = [
            ("mean", float(np.mean(o)), float(np.mean(r))),
            ("std", float(np.std(o)), float(np.std(r))),
            ("min", float(np.min(o)), float(np.min(r))),
            ("max", float(np.max(o)), float(np.max(r))),
            ("median", float(np.median(o)), float(np.median(r))),
        ]
        for name, o_val, r_val in pairs:
            stats[name] = StatComparison(
                original=o_val, replica=r_val, diff=abs(o_val - r_val)
            )

    o_null = orig.null_count() / max(len(orig), 1)
    r_null = repl.null_count() / max(len(repl), 1)
    stats["null_fraction"] = StatComparison(
        original=o_null, replica=r_null, diff=abs(o_null - r_null)
    )

    test = None
    if len(o) > 5 and len(r) > 5:
        ks_stat, p_value = sp_stats.ks_2samp(o, r)
        test = TestResult(
            test_name="ks_2samp", statistic=float(ks_stat), p_value=float(p_value)
        )

    return ColumnDiff(
        name=orig.name, dtype=dtype_str, stats_comparison=stats, distribution_test=test
    )


def _diff_boolean(orig: pl.Series, repl: pl.Series, dtype_str: str) -> ColumnDiff:
    """Compare boolean columns."""
    o_nn = orig.drop_nulls()
    r_nn = repl.drop_nulls()

    o_frac = float(o_nn.sum()) / max(len(o_nn), 1)
    r_frac = float(r_nn.sum()) / max(len(r_nn), 1)

    stats: dict[str, StatComparison] = {
        "true_fraction": StatComparison(
            original=o_frac, replica=r_frac, diff=abs(o_frac - r_frac)
        ),
        "null_fraction": StatComparison(
            original=orig.null_count() / max(len(orig), 1),
            replica=repl.null_count() / max(len(repl), 1),
            diff=abs(
                orig.null_count() / max(len(orig), 1)
                - repl.null_count() / max(len(repl), 1)
            ),
        ),
    }

    return ColumnDiff(name=orig.name, dtype=dtype_str, stats_comparison=stats)


def _diff_categorical(orig: pl.Series, repl: pl.Series, dtype_str: str) -> ColumnDiff:
    """Compare categorical/string columns."""
    o_nn = orig.drop_nulls()
    r_nn = repl.drop_nulls()

    stats: dict[str, StatComparison] = {
        "cardinality": StatComparison(
            original=o_nn.n_unique(),
            replica=r_nn.n_unique(),
            diff=abs(o_nn.n_unique() - r_nn.n_unique()),
        ),
        "null_fraction": StatComparison(
            original=orig.null_count() / max(len(orig), 1),
            replica=repl.null_count() / max(len(repl), 1),
            diff=abs(
                orig.null_count() / max(len(orig), 1)
                - repl.null_count() / max(len(repl), 1)
            ),
        ),
    }

    # Chi-squared test on shared categories
    test = None
    if len(o_nn) > 5 and len(r_nn) > 5:
        o_vc = o_nn.value_counts().sort(orig.name)
        r_vc = r_nn.value_counts().sort(orig.name)
        shared_labels = set(o_vc[orig.name].to_list()) & set(r_vc[orig.name].to_list())

        if len(shared_labels) > 1:
            labels = sorted(shared_labels)
            o_counts = np.array(
                [
                    o_vc.filter(pl.col(orig.name) == lbl)["count"][0]
                    if lbl in o_vc[orig.name].to_list()
                    else 0
                    for lbl in labels
                ],
                dtype=np.float64,
            )
            r_counts = np.array(
                [
                    r_vc.filter(pl.col(orig.name) == lbl)["count"][0]
                    if lbl in r_vc[orig.name].to_list()
                    else 0
                    for lbl in labels
                ],
                dtype=np.float64,
            )

            # Normalize to same total for comparison
            o_expected = o_counts / o_counts.sum() * r_counts.sum()
            valid = o_expected > 0
            if valid.sum() > 1:
                stat, p_value = sp_stats.chisquare(r_counts[valid], o_expected[valid])
                test = TestResult(
                    test_name="chisquare", statistic=float(stat), p_value=float(p_value)
                )

    return ColumnDiff(
        name=orig.name, dtype=dtype_str, stats_comparison=stats, distribution_test=test
    )


def _diff_date(orig: pl.Series, repl: pl.Series, dtype_str: str) -> ColumnDiff:
    """Compare date columns."""
    o_nn = orig.drop_nulls()
    r_nn = repl.drop_nulls()

    stats: dict[str, StatComparison] = {
        "min": StatComparison(
            original=str(o_nn.min()) if len(o_nn) > 0 else None,
            replica=str(r_nn.min()) if len(r_nn) > 0 else None,
        ),
        "max": StatComparison(
            original=str(o_nn.max()) if len(o_nn) > 0 else None,
            replica=str(r_nn.max()) if len(r_nn) > 0 else None,
        ),
        "null_fraction": StatComparison(
            original=orig.null_count() / max(len(orig), 1),
            replica=repl.null_count() / max(len(repl), 1),
            diff=abs(
                orig.null_count() / max(len(orig), 1)
                - repl.null_count() / max(len(repl), 1)
            ),
        ),
    }

    return ColumnDiff(name=orig.name, dtype=dtype_str, stats_comparison=stats)
