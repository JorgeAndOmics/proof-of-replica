"""CLI command: por profile-merge — merge profiles from multiple sources."""

from pathlib import Path

import click
import numpy as np

from proof_of_replica.core.column_schema import (
    CategoricalStats,
    ColumnDefinition,
    NumericStats,
)
from proof_of_replica.core.schema import Profile, load_profile, save_profile


@click.command("profile-merge")
@click.argument(
    "profiles", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--weights",
    type=float,
    multiple=True,
    default=None,
    help="Relative weights for each profile (space-separated).",
)
def profile_merge_cmd(
    profiles: tuple[Path, ...],
    output: Path,
    weights: tuple[float, ...] | None,
) -> None:
    """Merge profiles from multiple sources."""
    if len(profiles) < 2:
        msg = "At least 2 profiles are required."
        raise click.UsageError(msg)

    loaded = [load_profile(p) for p in profiles]

    w: list[float]
    if weights:
        if len(weights) != len(loaded):
            msg = f"Number of weights ({len(weights)}) must match profiles ({len(loaded)})."
            raise click.UsageError(msg)
        w = list(weights)
    else:
        w = [1.0 / len(loaded)] * len(loaded)

    # Normalize weights
    total = sum(w)
    w = [x / total for x in w]

    merged = _merge_profiles(loaded, w)
    save_profile(merged, output)
    click.echo(f"Merged {len(loaded)} profiles -> {output}")


def _merge_profiles(profiles: list[Profile], weights: list[float]) -> Profile:
    """Merge multiple profiles with weighted averaging."""
    # Collect all column names (union)
    all_cols: dict[str, list[tuple[ColumnDefinition, float]]] = {}
    for profile, weight in zip(profiles, weights, strict=True):
        for col in profile.columns:
            all_cols.setdefault(col.name, []).append((col, weight))

    merged_columns: list[ColumnDefinition] = []
    for col_name, entries in all_cols.items():
        merged_columns.append(_merge_column(col_name, entries))

    max_rows = max((p.row_count or 0) for p in profiles)

    return Profile(
        version=profiles[0].version,
        seed=profiles[0].seed,
        row_count=max_rows if max_rows > 0 else None,
        columns=merged_columns,
    )


def _merge_column(
    name: str, entries: list[tuple[ColumnDefinition, float]]
) -> ColumnDefinition:
    """Merge a single column across profiles."""
    # Use first entry's dtype and role
    first = entries[0][0]
    dtype = first.dtype

    # Merge stats by type
    merged_stats: NumericStats | CategoricalStats | None
    if isinstance(first.stats, NumericStats):
        merged_stats = _merge_numeric_stats(entries)
    elif isinstance(first.stats, CategoricalStats):
        merged_stats = _merge_categorical_stats(entries)
    else:
        merged_stats = None

    return ColumnDefinition(
        name=name,
        dtype=dtype,
        role=first.role,
        stats=merged_stats,
        distribution=first.distribution,
        generator=first.generator,
    )


def _merge_numeric_stats(
    entries: list[tuple[ColumnDefinition, float]],
) -> NumericStats:
    """Weighted average of numeric stats."""
    means: list[float] = []
    stds: list[float] = []
    weights: list[float] = []
    null_fracs: list[float] = []

    for col, w in entries:
        if not isinstance(col.stats, NumericStats):
            continue
        if col.stats.mean is not None:
            means.append(col.stats.mean)
            weights.append(w)
        if col.stats.std is not None:
            stds.append(col.stats.std)
        null_fracs.append(col.stats.null_fraction)

    w_arr = np.array(weights)
    w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr

    return NumericStats(
        mean=float(np.average(means, weights=w_arr)) if means else None,
        std=float(np.average(stds, weights=w_arr)) if stds else None,
        null_fraction=float(np.average(null_fracs, weights=w_arr))
        if null_fracs
        else 0.0,
    )


def _merge_categorical_stats(
    entries: list[tuple[ColumnDefinition, float]],
) -> CategoricalStats:
    """Union of categorical value counts with weighted averaging."""
    merged_vc: dict[str, float] = {}

    for col, w in entries:
        if not isinstance(col.stats, CategoricalStats):
            continue
        for label, frac in col.stats.value_counts.items():
            merged_vc[label] = merged_vc.get(label, 0.0) + frac * w

    # Normalize
    total = sum(merged_vc.values())
    if total > 0:
        merged_vc = {k: v / total for k, v in merged_vc.items()}

    return CategoricalStats(
        cardinality=len(merged_vc),
        value_counts=merged_vc,
    )
