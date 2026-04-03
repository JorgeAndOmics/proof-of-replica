"""CLI entry point for proof-of-replica."""

import click

from proof_of_replica import __version__
from proof_of_replica.cli.commands.generate_cmd import generate_cmd
from proof_of_replica.cli.commands.profile_cmd import profile
from proof_of_replica.cli.commands.validate_cmd import validate_cmd


@click.group()
@click.version_option(version=__version__, prog_name="por")
def app() -> None:
    """Proof of Replica — generate statistically faithful synthetic replicas of tabular datasets."""


app.add_command(profile)
app.add_command(generate_cmd, name="generate")
app.add_command(validate_cmd, name="validate")
