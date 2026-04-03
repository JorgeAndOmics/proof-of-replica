"""CLI command: por scaffold — generate a starter profile from a data file header."""

import logging
import sys
from pathlib import Path

import click

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import Profile, ProfilerConfig, save_profile
from proof_of_replica.engines._type_inference import infer_dtype
from proof_of_replica.utils.io import read_dataframe


@click.command("scaffold")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--rows",
    type=int,
    default=100,
    help="Rows to peek for type inference (0 = header only).",
)
@click.option("--separator", type=str, default=None, help="Column separator.")
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def scaffold(
    input_file: Path,
    output: Path | None,
    rows: int,
    separator: str | None,
    verbose: int,
) -> None:
    """Generate a starter profile from a data file header.

    Unlike 'profile', this does NOT compute real statistics —
    it only infers column types and outputs a skeleton profile
    for manual editing.
    """
    _configure_logging(verbose)

    n_rows = rows if rows > 0 else 1
    df = read_dataframe(input_file, separator=separator, n_rows=n_rows)
    config = ProfilerConfig()

    columns: list[ColumnDefinition] = []
    for col_name in df.columns:
        series = df[col_name]
        dtype = infer_dtype(series, len(df), config)
        columns.append(ColumnDefinition(name=col_name, dtype=dtype))

    profile = Profile(
        version="0.2.0",
        row_count=len(df),
        columns=columns,
    )

    output_path = output or input_file.with_suffix(".scaffold.json")
    save_profile(profile, output_path)

    if verbose > 0:
        click.echo(f"Scaffold saved to {output_path} ({len(columns)} columns)")


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
