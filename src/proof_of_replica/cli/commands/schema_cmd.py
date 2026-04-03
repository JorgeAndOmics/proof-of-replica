"""CLI command: por schema — output the profile JSON Schema."""

import json
from pathlib import Path

import click

from proof_of_replica.core.schema import generate_json_schema


@click.command("schema")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
def schema(output: Path | None) -> None:
    """Output the profile JSON Schema for editor integration."""
    schema_dict = generate_json_schema()
    text = json.dumps(schema_dict, indent=2) + "\n"

    if output is not None:
        output.write_text(text, encoding="utf-8")
        click.echo(f"Schema written to {output}")
    else:
        click.echo(text)
