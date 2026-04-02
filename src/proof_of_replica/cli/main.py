"""CLI entry point for proof-of-replica."""

import click

from proof_of_replica import __version__


@click.group()
@click.version_option(version=__version__, prog_name="por")
def app() -> None:
    """Proof of Replica — generate statistically faithful synthetic replicas of tabular datasets."""
