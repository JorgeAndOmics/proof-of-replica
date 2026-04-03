"""Tests for por diff CLI command."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def data_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Create original and replica CSV files."""
    rng = np.random.default_rng(42)
    orig = pl.DataFrame(
        {
            "x": rng.normal(50, 10, 100).tolist(),
            "label": (["A"] * 50 + ["B"] * 50),
        }
    )
    repl = pl.DataFrame(
        {
            "x": rng.normal(50, 10, 100).tolist(),
            "label": (["A"] * 45 + ["B"] * 55),
        }
    )
    orig_path = tmp_path / "original.csv"
    repl_path = tmp_path / "replica.csv"
    orig.write_csv(orig_path)
    repl.write_csv(repl_path)
    return orig_path, repl_path


class TestDiffCommand:
    def test_basic_text(self, runner, data_pair):
        orig, repl = data_pair
        result = runner.invoke(app, ["diff", str(orig), str(repl)])
        assert result.exit_code == 0
        assert "Column: x" in result.output

    def test_json_format(self, runner, data_pair):
        orig, repl = data_pair
        result = runner.invoke(app, ["diff", str(orig), str(repl), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "column_diffs" in data

    def test_privacy_flag(self, runner, data_pair):
        orig, repl = data_pair
        result = runner.invoke(app, ["diff", str(orig), str(repl), "--privacy"])
        assert result.exit_code == 0
        assert "Privacy distance" in result.output

    def test_report_file(self, runner, data_pair, tmp_path):
        orig, repl = data_pair
        report = tmp_path / "diff.txt"
        result = runner.invoke(
            app, ["diff", str(orig), str(repl), "--report", str(report)]
        )
        assert result.exit_code == 0
        assert report.exists()
        assert "Column: x" in report.read_text()
