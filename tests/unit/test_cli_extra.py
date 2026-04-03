"""Tests for CLI commands (template, scaffold, replicate)."""

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
def sample_csv(tmp_path: Path) -> Path:
    """Create a small CSV with mixed column types."""
    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "age": rng.normal(50, 10, size=100).tolist(),
            "group": (["A"] * 50 + ["B"] * 50),
            "active": ([True, False] * 50),
        }
    )
    path = tmp_path / "data.csv"
    df.write_csv(path)
    return path


class TestTemplateCommand:
    def test_minimal_template_stdout(self, runner):
        result = runner.invoke(app, ["template", "--minimal"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["version"] == "0.2.0"
        assert len(data["columns"]) == 2

    def test_full_template_stdout(self, runner):
        result = runner.invoke(app, ["template", "--full"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["columns"]) >= 5

    def test_default_template_stdout(self, runner):
        result = runner.invoke(app, ["template"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "columns" in data

    def test_template_to_file(self, runner, tmp_path):
        output = tmp_path / "template.json"
        result = runner.invoke(app, ["template", "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        assert "Template written" in result.output


class TestScaffoldCommand:
    def test_basic_scaffold(self, runner, sample_csv, tmp_path):
        output = tmp_path / "scaffold.json"
        result = runner.invoke(app, ["scaffold", str(sample_csv), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data["columns"]) == 3
        col_names = [c["name"] for c in data["columns"]]
        assert "age" in col_names
        assert "group" in col_names

    def test_scaffold_default_output(self, runner, sample_csv):
        result = runner.invoke(app, ["scaffold", str(sample_csv)])
        assert result.exit_code == 0
        default_path = sample_csv.with_suffix(".scaffold.json")
        assert default_path.exists()

    def test_scaffold_verbose(self, runner, sample_csv, tmp_path):
        output = tmp_path / "scaffold.json"
        result = runner.invoke(
            app, ["scaffold", str(sample_csv), "-o", str(output), "-v"]
        )
        assert result.exit_code == 0
        assert "Scaffold saved" in result.output


class TestReplicateCommand:
    def test_basic_replicate(self, runner, sample_csv, tmp_path):
        output = tmp_path / "replica.parquet"
        result = runner.invoke(app, ["replicate", str(sample_csv), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.exists()
        df = pl.read_parquet(output)
        assert len(df) > 0

    def test_replicate_csv_output(self, runner, sample_csv, tmp_path):
        output = tmp_path / "replica.csv"
        result = runner.invoke(
            app, ["replicate", str(sample_csv), "-o", str(output), "--format", "csv"]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_replicate_save_profile(self, runner, sample_csv, tmp_path):
        output = tmp_path / "replica.parquet"
        profile_out = tmp_path / "saved.profile.json"
        result = runner.invoke(
            app,
            [
                "replicate",
                str(sample_csv),
                "-o",
                str(output),
                "--save-profile",
                str(profile_out),
            ],
        )
        assert result.exit_code == 0
        assert profile_out.exists()
        data = json.loads(profile_out.read_text())
        assert data["version"] == "0.2.0"

    def test_replicate_verbose(self, runner, sample_csv, tmp_path):
        output = tmp_path / "replica.parquet"
        result = runner.invoke(
            app, ["replicate", str(sample_csv), "-o", str(output), "-v"]
        )
        assert result.exit_code == 0
        assert "Generated" in result.output

    def test_replicate_default_output(self, runner, sample_csv):
        result = runner.invoke(app, ["replicate", str(sample_csv)])
        assert result.exit_code == 0
        expected = sample_csv.with_name("data.replica.parquet")
        assert expected.exists()
