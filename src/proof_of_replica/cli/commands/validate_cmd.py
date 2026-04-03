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
@click.option(
    "--format", "fmt", type=click.Choice(["text", "json", "html"]), default="text"
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write report to file.",
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
def validate_cmd(
    replica: Path,
    profile_path: Path,
    strictness: str,
    fmt: str,
    report_path: Path | None,
    verbose: int,
) -> None:
    """Check replica fidelity against a profile."""
    _configure_logging(verbose)

    profile = load_profile(profile_path)
    df = read_dataframe(replica)
    result = validate(df, profile)

    output = _format_output(result, fmt)

    if report_path is not None:
        report_path.write_text(output, encoding="utf-8")
        if verbose > 0:
            click.echo(f"Report written to {report_path}")
    else:
        click.echo(output)

    if strictness == "fail" and not result.passed:
        raise SystemExit(1)


def _format_output(result: ValidationResult, fmt: str) -> str:
    """Format validation result as text, json, or html."""
    if fmt == "json":
        return result.model_dump_json(indent=2)
    if fmt == "html":
        return _format_html(result)
    return _format_text(result)


def _format_text(result: ValidationResult) -> str:
    """Format validation result as human-readable text."""
    lines: list[str] = []
    for check in result.checks:
        marker = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[check.status.value]
        col_str = f" [{check.column}]" if check.column else ""
        lines.append(f"  [{marker}]{col_str} {check.check_name}: {check.message}")

    lines.append(f"\nSummary: {result.summary}")
    lines.append("Result: PASSED" if result.passed else "Result: FAILED")
    return "\n".join(lines)


def _format_html(result: ValidationResult) -> str:
    """Format validation result as a simple HTML report."""
    rows = ""
    for check in result.checks:
        color = {"pass": "#2d7d46", "warn": "#b8860b", "fail": "#c0392b"}[
            check.status.value
        ]
        col = check.column or ""
        rows += (
            f"<tr><td style='color:{color};font-weight:bold'>{check.status.value.upper()}</td>"
            f"<td>{col}</td><td>{check.check_name}</td><td>{check.message}</td></tr>\n"
        )

    status_color = "#2d7d46" if result.passed else "#c0392b"
    status_text = "PASSED" if result.passed else "FAILED"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Validation Report</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f5f5f5}}</style>
</head><body>
<h1>Validation Report</h1>
<p>Result: <span style="color:{status_color};font-weight:bold">{status_text}</span></p>
<p>Summary: {result.summary}</p>
<table><tr><th>Status</th><th>Column</th><th>Check</th><th>Message</th></tr>
{rows}</table>
</body></html>"""


def _configure_logging(verbose: int) -> None:
    """Set logging level based on verbosity."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:  # pragma: no cover
        level = logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr)
