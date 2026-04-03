"""CLI command: por diff — compare original data against a replica."""

import logging
import sys
from pathlib import Path

import click

from proof_of_replica.engines.diff import DiffResult, diff_datasets
from proof_of_replica.utils.io import read_dataframe


@click.command("diff")
@click.argument("original", type=click.Path(exists=True, path_type=Path))
@click.argument("replica", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write report to file.",
)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.option("--privacy", is_flag=True, help="Include privacy distance metrics.")
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def diff_cmd(
    original: Path,
    replica: Path,
    report_path: Path | None,
    fmt: str,
    privacy: bool,
    verbose: int,
) -> None:
    """Compare original data against a synthetic replica."""
    _configure_logging(verbose)

    orig_df = read_dataframe(original)
    repl_df = read_dataframe(replica)

    result = diff_datasets(orig_df, repl_df, privacy=privacy)

    output = _format_output(result, fmt)

    if report_path is not None:
        report_path.write_text(output, encoding="utf-8")
        if verbose > 0:
            click.echo(f"Diff report written to {report_path}")
    else:
        click.echo(output)


def _format_output(result: DiffResult, fmt: str) -> str:
    """Format diff result as text or json."""
    if fmt == "json":
        return result.model_dump_json(indent=2)
    return _format_text(result)


def _format_text(result: DiffResult) -> str:
    """Format diff result as human-readable text."""
    lines: list[str] = []

    for col_diff in result.column_diffs:
        lines.append(f"\n  Column: {col_diff.name} ({col_diff.dtype})")
        for stat_name, comp in col_diff.stats_comparison.items():
            diff_str = f" (diff: {comp.diff:.4f})" if comp.diff is not None else ""
            lines.append(
                f"    {stat_name}: original={comp.original}, replica={comp.replica}{diff_str}"
            )
        if col_diff.distribution_test is not None:
            t = col_diff.distribution_test
            lines.append(
                f"    {t.test_name}: statistic={t.statistic:.4f}, p-value={t.p_value:.4f}"
            )

    if result.privacy is not None:
        lines.append("\n  Privacy distance:")
        for key, val in result.privacy.items():
            lines.append(f"    {key}: {val:.4f}")

    return "\n".join(lines)


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:  # pragma: no cover
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
