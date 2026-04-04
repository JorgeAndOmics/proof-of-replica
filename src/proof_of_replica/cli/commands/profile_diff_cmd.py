"""CLI command: por profile-diff — compare two profiles."""

import json
from pathlib import Path
from typing import Any

import click

from proof_of_replica.core.schema import Profile, load_profile


@click.command("profile-diff")
@click.argument("profile_a", type=click.Path(exists=True, path_type=Path))
@click.argument("profile_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write comparison to file.",
)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def profile_diff_cmd(
    profile_a: Path,
    profile_b: Path,
    report_path: Path | None,
    fmt: str,
) -> None:
    """Compare two profiles side by side."""
    pa = load_profile(profile_a)
    pb = load_profile(profile_b)

    diff = _compare_profiles(pa, pb)
    output = json.dumps(diff, indent=2) if fmt == "json" else _format_text(diff)

    if report_path is not None:
        report_path.write_text(output, encoding="utf-8")
    else:
        click.echo(output)


def _compare_profiles(pa: Profile, pb: Profile) -> dict[str, Any]:
    """Compare two profiles and return a diff dict."""

    diff: dict[str, Any] = {
        "metadata": {
            "a_version": pa.version,
            "b_version": pb.version,
            "a_row_count": pa.row_count,
            "b_row_count": pb.row_count,
            "a_seed": pa.seed,
            "b_seed": pb.seed,
        },
        "columns": [],
    }

    a_cols = {c.name: c for c in pa.columns}
    b_cols = {c.name: c for c in pb.columns}
    all_names = list(dict.fromkeys([*a_cols, *b_cols]))

    col_diffs = []
    for name in all_names:
        ca = a_cols.get(name)
        cb = b_cols.get(name)

        if ca is None:
            col_diffs.append({"name": name, "status": "only_in_b"})
        elif cb is None:
            col_diffs.append({"name": name, "status": "only_in_a"})
        else:
            col_diff: dict[str, Any] = {"name": name, "status": "both"}
            if ca.dtype != cb.dtype:
                col_diff["dtype_a"] = ca.dtype.value
                col_diff["dtype_b"] = cb.dtype.value

            # Compare stats
            if ca.stats is not None and cb.stats is not None:
                a_dump = ca.stats.model_dump()
                b_dump = cb.stats.model_dump()
                stat_diffs = {}
                for key in set(a_dump) | set(b_dump):
                    av = a_dump.get(key)
                    bv = b_dump.get(key)
                    if av != bv:
                        stat_diffs[key] = {"a": av, "b": bv}
                if stat_diffs:
                    col_diff["stats_diff"] = stat_diffs

            col_diffs.append(col_diff)

    diff["columns"] = col_diffs
    return diff


def _format_text(diff: dict[str, Any]) -> str:
    """Format profile diff as human-readable text."""
    lines: list[str] = []

    meta = diff["metadata"]
    lines.append(
        f"Profile A: v{meta['a_version']}, {meta['a_row_count']} rows, seed={meta['a_seed']}"
    )
    lines.append(
        f"Profile B: v{meta['b_version']}, {meta['b_row_count']} rows, seed={meta['b_seed']}"
    )
    lines.append("")

    for col in diff["columns"]:
        status = col["status"]
        if status == "only_in_a":
            lines.append(f"  {col['name']}: only in A")
        elif status == "only_in_b":
            lines.append(f"  {col['name']}: only in B")
        else:
            dtype_note = ""
            if "dtype_a" in col:
                dtype_note = f" (dtype: {col['dtype_a']} vs {col['dtype_b']})"
            stat_diffs = col.get("stats_diff", {})
            if stat_diffs or dtype_note:
                lines.append(f"  {col['name']}:{dtype_note}")
                for key, vals in stat_diffs.items():
                    lines.append(f"    {key}: {vals['a']} -> {vals['b']}")
            else:
                lines.append(f"  {col['name']}: identical")

    return "\n".join(lines)
