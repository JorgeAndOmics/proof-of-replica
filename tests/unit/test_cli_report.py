"""Tests for por report CLI command."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import Profile, save_profile


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def report_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create profile, replica, and original files for report testing."""
    rng = np.random.default_rng(42)

    profile = Profile(
        version="0.2.0",
        seed=42,
        row_count=50,
        columns=[
            ColumnDefinition(
                name="x",
                dtype="float64",
                stats={"mean": 0.0, "std": 1.0},
                distribution={"family": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ),
        ],
    )
    profile_path = tmp_path / "profile.json"
    save_profile(profile, profile_path)

    replica = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
    replica_path = tmp_path / "replica.parquet"
    replica.write_parquet(replica_path)

    original = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
    original_path = tmp_path / "original.parquet"
    original.write_parquet(original_path)

    return profile_path, replica_path, original_path


class TestReportCommand:
    def test_basic_report(self, runner, report_fixtures, tmp_path):
        profile_path, replica_path, _ = report_fixtures
        output = tmp_path / "report.html"
        result = runner.invoke(
            app, ["report", str(replica_path), str(profile_path), "-o", str(output)]
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Fidelity Report" in content

    def test_report_with_original(self, runner, report_fixtures, tmp_path):
        profile_path, replica_path, original_path = report_fixtures
        output = tmp_path / "report.html"
        result = runner.invoke(
            app,
            [
                "report",
                str(replica_path),
                str(profile_path),
                "-o",
                str(output),
                "--original",
                str(original_path),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_report_verbose(self, runner, report_fixtures, tmp_path):
        profile_path, replica_path, _ = report_fixtures
        output = tmp_path / "report.html"
        result = runner.invoke(
            app,
            ["report", str(replica_path), str(profile_path), "-o", str(output), "-v"],
        )
        assert result.exit_code == 0
        assert "Report written" in result.output
