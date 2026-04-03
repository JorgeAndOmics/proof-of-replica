"""CLI command: por profile — extract a statistical profile from real data."""

import logging
import sys
from pathlib import Path

import click

from proof_of_replica.core.schema import ProfilerConfig, save_profile
from proof_of_replica.engines.profiling import profile_dataframe
from proof_of_replica.utils.io import read_dataframe


@click.command("profile")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--separator", type=str, default=None, help="Column separator for CSV/TSV."
)
@click.option("--rows", type=int, default=0, help="Rows to sample (0 = all).")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--correlations", is_flag=True, help="Compute correlation matrix.")
@click.option("--group-column", type=str, default=None, help="Group column name.")
@click.option("--dp", is_flag=True, help="Enable differential privacy mode.")
@click.option("--epsilon", type=float, default=1.0, help="DP epsilon parameter.")
@click.option("--delta", type=float, default=1e-5, help="DP delta parameter.")
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def profile(
    input_file: Path,
    output: Path | None,
    separator: str | None,
    rows: int,
    seed: int,
    correlations: bool,
    group_column: str | None,
    dp: bool,
    epsilon: float,
    delta: float,
    verbose: int,
) -> None:
    """Extract a statistical profile from real data."""
    _configure_logging(verbose)

    n_rows = rows if rows > 0 else None
    df = read_dataframe(input_file, separator=separator, n_rows=n_rows)

    result = profile_dataframe(
        df,
        correlations=correlations,
        group_column=group_column,
        seed=seed,
        config=ProfilerConfig(),
        dp=dp,
        epsilon=epsilon,
        delta=delta,
    )

    output_path = output or input_file.with_suffix(".profile.json")
    save_profile(result, output_path)

    if verbose > 0:
        click.echo(
            f"Profile saved to {output_path} ({len(result.columns)} columns, {result.row_count} rows)"
        )


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:  # pragma: no cover
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
