"""HTML fidelity report generation."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.enums import ColumnDtype
from proof_of_replica.core.schema import Profile
from proof_of_replica.engines._report_plots import (
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_numeric_distribution,
)
from proof_of_replica.engines.validation import ValidationResult, validate

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2em; color: #333; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
h2 { color: #34495e; margin-top: 1.5em; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background: #f5f5f5; font-weight: 600; }
tr:nth-child(even) { background: #fafafa; }
.pass { color: #27ae60; font-weight: bold; }
.warn { color: #f39c12; font-weight: bold; }
.fail { color: #e74c3c; font-weight: bold; }
.plot { text-align: center; margin: 1em 0; }
img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
"""


def generate_fidelity_report(
    replica: pl.DataFrame,
    profile: Profile,
    *,
    original: pl.DataFrame | None = None,
) -> str:
    """Generate an HTML fidelity report.

    Args:
        replica: The synthetic replica DataFrame.
        profile: The profile used for generation.
        original: Optional original data for side-by-side comparison.

    Returns:
        Self-contained HTML string.
    """
    result = validate(replica, profile)
    sections: list[str] = []

    # Summary section
    sections.append(_build_summary_section(result))

    # Per-column sections
    for col_def in profile.columns:
        if col_def.name not in replica.columns:
            continue
        repl_series = replica[col_def.name]
        orig_series = (
            original[col_def.name]
            if original is not None and col_def.name in original.columns
            else None
        )
        sections.append(
            _build_column_section(col_def, repl_series, orig_series, result)
        )

    # Correlation section
    if profile.correlations is not None:
        sections.append(_build_correlation_section(replica, profile, original))

    return _render_html(sections)


def generate_validation_report(result: ValidationResult) -> str:
    """Generate an HTML report from validation results.

    Args:
        result: Validation result to render.

    Returns:
        Self-contained HTML string.
    """
    sections = [_build_summary_section(result)]
    return _render_html(sections)


# ── Section builders ─────────────────────────────────────────


def _build_summary_section(result: ValidationResult) -> str:
    """Build the overall summary table."""
    status_class = "pass" if result.passed else "fail"
    status_text = "PASSED" if result.passed else "FAILED"

    rows = ""
    for check in result.checks:
        col = check.column or "—"
        rows += (
            f"<tr><td class='{check.status.value}'>{check.status.value.upper()}</td>"
            f"<td>{col}</td><td>{check.check_name}</td><td>{check.message}</td></tr>\n"
        )

    return f"""
    <h2>Fidelity Summary</h2>
    <p>Overall: <span class="{status_class}">{status_text}</span>
    (pass: {result.summary.get("pass", 0)}, warn: {result.summary.get("warn", 0)},
    fail: {result.summary.get("fail", 0)})</p>
    <table><tr><th>Status</th><th>Column</th><th>Check</th><th>Message</th></tr>
    {rows}</table>
    """


def _build_column_section(
    col_def: ColumnDefinition,
    repl_series: pl.Series,
    orig_series: pl.Series | None,
    result: ValidationResult,
) -> str:
    """Build the section for a single column."""
    parts = [f"<h3>{col_def.name} ({col_def.dtype.value})</h3>"]

    # Distribution plot
    if col_def.dtype in (ColumnDtype.FLOAT64, ColumnDtype.INT64):
        img = plot_numeric_distribution(
            repl_series, original=orig_series, title=col_def.name
        )
        parts.append(f'<div class="plot"><img src="data:image/png;base64,{img}"></div>')
    elif col_def.dtype == ColumnDtype.CATEGORICAL:
        img = plot_categorical_distribution(
            repl_series, original=orig_series, title=col_def.name
        )
        parts.append(f'<div class="plot"><img src="data:image/png;base64,{img}"></div>')

    # Column-specific checks
    col_checks = [c for c in result.checks if c.column == col_def.name]
    if col_checks:
        rows = ""
        for c in col_checks:
            rows += f"<tr><td class='{c.status.value}'>{c.status.value.upper()}</td><td>{c.check_name}</td><td>{c.message}</td></tr>\n"
        parts.append(
            f"<table><tr><th>Status</th><th>Check</th><th>Message</th></tr>{rows}</table>"
        )

    return "\n".join(parts)


def _build_correlation_section(
    replica: pl.DataFrame,
    profile: Profile,
    original: pl.DataFrame | None,
) -> str:
    """Build the correlation heatmap section."""
    if profile.correlations is None:
        return ""

    cols = profile.correlations.columns
    available = [c for c in cols if c in replica.columns]
    if len(available) < 2:
        return ""

    parts = ["<h2>Correlation Structure</h2>"]

    # Replica correlation
    repl_arr = (
        replica.select(available)
        .cast(dict.fromkeys(available, pl.Float64))
        .drop_nulls()
        .to_numpy()
    )
    if len(repl_arr) > 2:
        repl_corr = np.corrcoef(repl_arr, rowvar=False)
        img = plot_correlation_heatmap(
            repl_corr, available, title="Replica Correlation"
        )
        parts.append(f'<div class="plot"><img src="data:image/png;base64,{img}"></div>')

    # Original correlation if available
    if original is not None:
        orig_available = [c for c in available if c in original.columns]
        if len(orig_available) >= 2:
            orig_arr = (
                original.select(orig_available)
                .cast(dict.fromkeys(orig_available, pl.Float64))
                .drop_nulls()
                .to_numpy()
            )
            if len(orig_arr) > 2:
                orig_corr = np.corrcoef(orig_arr, rowvar=False)
                img = plot_correlation_heatmap(
                    orig_corr, orig_available, title="Original Correlation"
                )
                parts.append(
                    f'<div class="plot"><img src="data:image/png;base64,{img}"></div>'
                )

    return "\n".join(parts)


def _render_html(sections: list[str]) -> str:
    """Wrap sections in a self-contained HTML document."""
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Proof of Replica — Fidelity Report</title>
<style>{_CSS}</style></head>
<body>
<h1>Proof of Replica — Fidelity Report</h1>
{body}
</body></html>"""
