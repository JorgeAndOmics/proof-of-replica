"""CLI command: por generate — generate a synthetic replica from a profile."""

import json
import logging
import sys
from pathlib import Path

import click

from proof_of_replica.core.schema import load_profile, merge_overrides
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
    validate_mode: str,
    verbose: int,
) -> None:
    """Generate a synthetic replica from a profile."""
    _configure_logging(verbose)

    profile = load_profile(profile_path)

    # Apply overrides
    overrides = _load_overrides(override_json, override_file)
    if overrides:
        profile = merge_overrides(profile, overrides)

    df = generate(profile, row_count=rows, seed=seed)

    output_path = output or Path(f"replica.{fmt}")
    write_dataframe(df, output_path)

    if verbose > 0:
        click.echo(f"Generated {len(df)} rows -> {output_path}")

    if validate_mode != "off":
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
