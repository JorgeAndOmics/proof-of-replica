"""CLI command: por generate — generate a synthetic replica from a profile."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from proof_of_replica.core.schema import Profile, load_profile, merge_overrides

if TYPE_CHECKING:
    import polars as pl
from proof_of_replica.engines.generation import generate
from proof_of_replica.engines.validation import validate
from proof_of_replica.utils.io import write_dataframe


@click.command("generate")
@click.argument("profile_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--format", "fmt", type=click.Choice(["csv", "tsv", "parquet"]), default="parquet"
)
@click.option("--rows", type=int, default=None, help="Override row count.")
@click.option("--seed", type=int, default=None, help="Override seed.")
@click.option(
    "--override", "override_json", type=str, default=None, help="Inline JSON overrides."
)
@click.option(
    "--override-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON overrides file.",
)
@click.option(
    "--variants",
    type=int,
    default=None,
    help="Generate N variants with incremented seeds.",
)
@click.option(
    "--split", type=float, default=None, help="Train/test split ratio (e.g. 0.8)."
)
@click.option(
    "--validate",
    "validate_mode",
    type=click.Choice(["off", "warn", "fail"]),
    default="warn",
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def generate_cmd(
    profile_path: Path,
    output: Path | None,
    fmt: str,
    rows: int | None,
    seed: int | None,
    override_json: str | None,
    override_file: Path | None,
    variants: int | None,
    split: float | None,
    validate_mode: str,
    verbose: int,
) -> None:
    """Generate a synthetic replica from a profile."""
    _configure_logging(verbose)

    if variants is not None and split is not None:
        msg = "--variants and --split are mutually exclusive."
        raise click.UsageError(msg)

    profile = load_profile(profile_path)

    # Apply overrides
    overrides = _load_overrides(override_json, override_file)
    if overrides:
        profile = merge_overrides(profile, overrides)

    base_seed = seed if seed is not None else profile.seed

    if variants is not None:
        _generate_variants(
            profile, variants, base_seed, rows, fmt, output, validate_mode, verbose
        )
    elif split is not None:
        _generate_split(
            profile, split, base_seed, rows, fmt, output, validate_mode, verbose
        )
    else:
        _generate_single(profile, base_seed, rows, fmt, output, validate_mode, verbose)


def _generate_single(
    profile: Profile,
    seed: int,
    rows: int | None,
    fmt: str,
    output: Path | None,
    validate_mode: str,
    verbose: int,
) -> None:
    """Generate a single replica."""
    df = generate(profile, row_count=rows, seed=seed)
    output_path = output or Path(f"replica.{fmt}")
    write_dataframe(df, output_path)

    if verbose > 0:
        click.echo(f"Generated {len(df)} rows -> {output_path}")

    _run_validation(df, profile, validate_mode, verbose)


def _generate_variants(
    profile: Profile,
    n: int,
    base_seed: int,
    rows: int | None,
    fmt: str,
    output: Path | None,
    validate_mode: str,
    verbose: int,
) -> None:
    """Generate N variant replicas with incremented seeds."""
    base = output or Path(f"replica.{fmt}")
    stem = base.stem
    suffix = base.suffix or f".{fmt}"
    parent = base.parent

    for i in range(n):
        variant_seed = base_seed + i
        df = generate(profile, row_count=rows, seed=variant_seed)
        variant_path = parent / f"{stem}_{i + 1:03d}{suffix}"
        write_dataframe(df, variant_path)

        if verbose > 0:
            click.echo(f"Variant {i + 1}/{n}: {len(df)} rows -> {variant_path}")

        _run_validation(df, profile, validate_mode, verbose)


def _generate_split(
    profile: Profile,
    ratio: float,
    seed: int,
    rows: int | None,
    fmt: str,
    output: Path | None,
    validate_mode: str,
    verbose: int,
) -> None:
    """Generate a single DataFrame and split into train/test."""
    df = generate(profile, row_count=rows, seed=seed)
    split_idx = int(len(df) * ratio)

    train_df = df.head(split_idx)
    test_df = df.tail(len(df) - split_idx)

    base = output or Path(f"replica.{fmt}")
    stem = base.stem
    suffix = base.suffix or f".{fmt}"
    parent = base.parent

    train_path = parent / f"{stem}_train{suffix}"
    test_path = parent / f"{stem}_test{suffix}"

    write_dataframe(train_df, train_path)
    write_dataframe(test_df, test_path)

    if verbose > 0:
        click.echo(f"Train: {len(train_df)} rows -> {train_path}")
        click.echo(f"Test: {len(test_df)} rows -> {test_path}")

    _run_validation(df, profile, validate_mode, verbose)


def _run_validation(
    df: pl.DataFrame,
    profile: Profile,
    validate_mode: str,
    verbose: int,
) -> None:
    """Run validation if enabled."""
    if validate_mode == "off":
        return

    result = validate(df, profile)
    if verbose > 0 or not result.passed:
        for check in result.checks:
            if check.status != "pass":
                click.echo(
                    f"  [{check.status.upper()}] {check.check_name}: {check.message}",
                    err=True,
                )

    if validate_mode == "fail" and not result.passed:
        raise SystemExit(1)


def _load_overrides(
    override_json: str | None, override_file: Path | None
) -> dict[str, object]:
    """Parse override sources into a single dict."""
    result: dict[str, object] = {}

    if override_file is not None:
        text = override_file.read_text(encoding="utf-8")
        file_data = json.loads(text)
        if isinstance(file_data, dict):
            result.update(file_data)

    if override_json is not None:
        inline_data = json.loads(override_json)
        if isinstance(inline_data, dict):
            result.update(inline_data)

    return result


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:  # pragma: no cover
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
