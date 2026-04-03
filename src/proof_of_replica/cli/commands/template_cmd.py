"""CLI command: por template — output an annotated example profile."""

import json
from pathlib import Path

import click

from proof_of_replica.core.schema import Profile

_MINIMAL_PROFILE: dict[str, object] = {
    "version": "0.2.0",
    "seed": 42,
    "row_count": 100,
    "columns": [
        {
            "name": "value",
            "dtype": "float64",
            "stats": {"mean": 50.0, "std": 10.0, "min": 18.0, "max": 90.0},
            "distribution": {
                "family": "normal",
                "params": {"loc": 50.0, "scale": 10.0},
            },
        },
        {
            "name": "category",
            "dtype": "categorical",
            "stats": {"cardinality": 3, "value_counts": {"A": 0.5, "B": 0.3, "C": 0.2}},
        },
    ],
}

_FULL_PROFILE: dict[str, object] = {
    "version": "0.2.0",
    "seed": 42,
    "row_count": 500,
    "columns": [
        {
            "name": "sample_id",
            "dtype": "string",
            "role": "identifier",
            "generator": {"method": "sequential", "prefix": "S_", "zero_pad": 4},
        },
        {
            "name": "age",
            "dtype": "float64",
            "stats": {
                "mean": 52.3,
                "std": 14.7,
                "min": 18.0,
                "max": 89.0,
                "null_fraction": 0.02,
            },
            "distribution": {
                "family": "normal",
                "params": {"loc": 52.3, "scale": 14.7},
            },
            "constraints": {"min": 18.0, "max": 100.0},
        },
        {
            "name": "condition",
            "dtype": "categorical",
            "role": "group",
            "stats": {
                "cardinality": 3,
                "value_counts": {"healthy": 0.40, "stage_1": 0.35, "stage_2": 0.25},
            },
        },
        {"name": "is_control", "dtype": "boolean", "stats": {"true_fraction": 0.40}},
        {
            "name": "measurement_date",
            "dtype": "date",
            "stats": {"min": "2020-01-15", "max": "2025-11-30"},
        },
        {
            "name": "sequence",
            "dtype": "string",
            "generator": {
                "method": "alphabet",
                "chars": "ACGT",
                "length": {"mean": 150, "std": 20, "min": 50},
            },
        },
    ],
    "correlations": {"columns": ["age"], "matrix": [[1.0]]},
    "missingness": {"pattern": "MCAR"},
}


@click.command("template")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--minimal", is_flag=True, help="Minimal profile with required fields only."
)
@click.option("--full", is_flag=True, help="Comprehensive profile with all sections.")
def template(output: Path | None, minimal: bool, full: bool) -> None:  # noqa: ARG001
    """Output an annotated example profile JSON."""
    raw = _FULL_PROFILE if full else _MINIMAL_PROFILE
    profile = Profile.model_validate(raw)

    data = profile.model_dump(mode="json", exclude_none=True)
    text = json.dumps(data, indent=2) + "\n"

    if output is not None:
        output.write_text(text, encoding="utf-8")
        click.echo(f"Template written to {output}")
    else:
        click.echo(text)
