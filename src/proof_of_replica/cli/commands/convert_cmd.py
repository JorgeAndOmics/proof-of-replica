"""CLI command: por convert — format conversion utility."""

from pathlib import Path

import click

from proof_of_replica.utils.io import read_dataframe, write_dataframe


@click.command("convert")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
def convert(input_file: Path, output: Path) -> None:
    """Convert between CSV, TSV, and Parquet formats."""
    df = read_dataframe(input_file)
    write_dataframe(df, output)
    click.echo(f"Converted {input_file} -> {output}")
