"""CLI command: por validate — check replica fidelity against a profile."""

import logging
import sys
from pathlib import Path

import click

from proof_of_replica.core.schema import load_profile
from proof_of_replica.engines.validation import ValidationResult, validate
from proof_of_replica.utils.io import read_dataframe


@click.command("validate")
@click.argument("replica", type=click.Path(exists=True, path_type=Path))
@click.argument("profile_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--strictness",
    type=click.Choice(["warn", "fail"]),
    default="warn",
)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def validate_cmd(
    replica: Path,
    profile_path: Path,
    strictness: str,
    fmt: str,
    verbose: int,
) -> None:
    """Check replica fidelity against a profile."""
    _configure_logging(verbose)

    profile = load_profile(profile_path)
    df = read_dataframe(replica)
    result = validate(df, profile)

    if fmt == "json":
        click.echo(result.model_dump_json(indent=2))
    else:
        _print_text_report(result)

    if strictness == "fail" and not result.passed:
        raise SystemExit(1)


def _print_text_report(result: ValidationResult) -> None:
    """Print a human-readable validation report."""
    for check in result.checks:
        marker = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[check.status.value]
        col_str = f" [{check.column}]" if check.column else ""
        click.echo(f"  [{marker}]{col_str} {check.check_name}: {check.message}")

    click.echo(f"\nSummary: {result.summary}")
    if result.passed:
        click.echo("Result: PASSED")
    else:
        click.echo("Result: FAILED")


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
