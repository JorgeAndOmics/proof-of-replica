"""CLI command: por report — generate an HTML fidelity report."""

import logging
import sys
from pathlib import Path

import click

from proof_of_replica.core.schema import load_profile
from proof_of_replica.engines.reporting import generate_fidelity_report
from proof_of_replica.utils.io import read_dataframe


@click.command("report")
@click.argument("replica", type=click.Path(exists=True, path_type=Path))
@click.argument("profile_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=Path("report.html")
)
@click.option("--original", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def report(
    replica: Path,
    profile_path: Path,
    output: Path,
    original: Path | None,
    verbose: int,
) -> None:
    """Generate an HTML fidelity report."""
    _configure_logging(verbose)

    profile = load_profile(profile_path)
    repl_df = read_dataframe(replica)
    orig_df = read_dataframe(original) if original is not None else None

    html = generate_fidelity_report(repl_df, profile, original=orig_df)
    output.write_text(html, encoding="utf-8")

    if verbose > 0:
        click.echo(f"Report written to {output}")


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:  # pragma: no cover
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
