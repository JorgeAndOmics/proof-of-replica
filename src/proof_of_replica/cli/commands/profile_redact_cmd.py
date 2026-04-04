"""CLI command: por profile-redact — strip sensitive info from a profile."""

import math
from pathlib import Path
from typing import Any

import click

from proof_of_replica.core.schema import Profile, load_profile, save_profile


@click.command("profile-redact")
@click.argument("profile_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option("--drop-columns", multiple=True, help="Column names to remove entirely.")
@click.option(
    "--drop-stats",
    multiple=True,
    help="Stat fields to remove (e.g. percentiles, min, max).",
)
@click.option(
    "--round-to",
    type=int,
    default=3,
    help="Round numeric stats to N significant figures.",
)
def profile_redact_cmd(
    profile_path: Path,
    output: Path,
    drop_columns: tuple[str, ...],
    drop_stats: tuple[str, ...],
    round_to: int,
) -> None:
    """Strip sensitive information from a profile."""
    profile = load_profile(profile_path)
    data = profile.model_dump(mode="json", exclude_none=True)

    # Drop columns
    if drop_columns:
        drop_set = set(drop_columns)
        data["columns"] = [c for c in data["columns"] if c["name"] not in drop_set]

    # Drop stat fields and round
    for col in data["columns"]:
        stats = col.get("stats")
        if stats is None:
            continue

        # Drop specified stat fields
        for field in drop_stats:
            stats.pop(field, None)

        # Round numeric values
        _round_dict(stats, round_to)

    # Re-validate
    redacted = Profile.model_validate(data)
    save_profile(redacted, output)
    click.echo(f"Redacted profile -> {output}")


def _round_dict(d: dict[str, Any], sig_figs: int) -> None:
    """Round all float values in a dict to N significant figures."""
    for key, val in d.items():
        if isinstance(val, float):
            d[key] = _round_sig(val, sig_figs)
        elif isinstance(val, dict):
            _round_dict(val, sig_figs)


def _round_sig(x: float, sig: int) -> float:
    """Round a float to N significant figures."""
    if x == 0:
        return 0.0
    return round(x, sig - 1 - math.floor(math.log10(abs(x))))
