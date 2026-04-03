"""CLI command: por replicate — one-command data-in, replica-out."""

import logging
import sys
from pathlib import Path

import click

from proof_of_replica.core.schema import DefaultsConfig, ProfilerConfig, save_profile
from proof_of_replica.engines.generation import generate
from proof_of_replica.engines.profiling import profile_dataframe
from proof_of_replica.utils.io import read_dataframe, write_dataframe


@click.command("replicate")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--format", "fmt", type=click.Choice(["csv", "tsv", "parquet"]), default="parquet"
)
@click.option(
    "--save-profile", "save_profile_path", type=click.Path(path_type=Path), default=None
)
@click.option("--rows", type=int, default=None, help="Override row count.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--correlations", is_flag=True, help="Preserve correlation structure.")
@click.option("--group-column", type=str, default=None, help="Group column name.")
@click.option("--noise-level", type=float, default=0.05, help="Global noise level.")
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def replicate(
    input_file: Path,
    output: Path | None,
    fmt: str,
    save_profile_path: Path | None,
    rows: int | None,
    seed: int,
    correlations: bool,
    group_column: str | None,
    noise_level: float,
    verbose: int,
) -> None:
    """One-command data-in, replica-out (auto mode).

    Profiles the data, generates a replica, and writes the output.
    """
    _configure_logging(verbose)

    df = read_dataframe(input_file)

    profile = profile_dataframe(
        df,
        correlations=correlations,
        group_column=group_column,
        seed=seed,
        config=ProfilerConfig(),
    )

    # Apply noise_level override
    if noise_level != profile.defaults.noise_level:
        profile = profile.model_copy(
            update={"defaults": DefaultsConfig(noise_level=noise_level)}
        )

    if save_profile_path is not None:
        save_profile(profile, save_profile_path)
        if verbose > 0:
            click.echo(f"Profile saved to {save_profile_path}")

    replica = generate(profile, row_count=rows, seed=seed)

    output_path = output or input_file.with_name(f"{input_file.stem}.replica.{fmt}")
    write_dataframe(replica, output_path)

    if verbose > 0:
        click.echo(
            f"Generated {len(replica)}-row replica with "
            f"{len(replica.columns)} columns -> {output_path}"
        )


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:  # pragma: no cover
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
