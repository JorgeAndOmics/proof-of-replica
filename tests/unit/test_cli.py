"""Tests for CLI commands (profile, generate, validate)."""

import json
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


@pytest.fixture
def sample_profile(tmp_path: Path) -> Path:
    """Create a minimal valid profile JSON."""
    p = Profile(
        version="0.2.0",
        seed=42,
        row_count=50,
        columns=[
            ColumnDefinition(
                name="x",
                dtype="float64",
                stats={"mean": 50.0, "std": 10.0},
                distribution={
                    "family": "normal",
                    "params": {"loc": 50.0, "scale": 10.0},
                },
            ),
        ],
    )
    path = tmp_path / "profile.json"
    save_profile(p, path)
    return path


class TestProfileCommand:
    def test_basic_profile(self, runner, sample_csv, tmp_path):
        output = tmp_path / "out.profile.json"
        result = runner.invoke(app, ["profile", str(sample_csv), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["version"] == "0.2.0"
        assert len(data["columns"]) == 3

    def test_profile_default_output(self, runner, sample_csv):
        result = runner.invoke(app, ["profile", str(sample_csv)])
        assert result.exit_code == 0
        default_path = sample_csv.with_suffix(".profile.json")
        assert default_path.exists()

    def test_profile_verbose(self, runner, sample_csv, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(
            app, ["profile", str(sample_csv), "-o", str(output), "-v"]
        )
        assert result.exit_code == 0
        assert "Profile saved" in result.output

    def test_profile_with_correlations(self, runner, sample_csv, tmp_path):
        output = tmp_path / "corr.json"
        result = runner.invoke(
            app, ["profile", str(sample_csv), "-o", str(output), "--correlations"]
        )
        assert result.exit_code == 0


class TestGenerateCommand:
    def test_basic_generate(self, runner, sample_profile, tmp_path):
        output = tmp_path / "replica.parquet"
        result = runner.invoke(
            app, ["generate", str(sample_profile), "-o", str(output)]
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_generate_csv(self, runner, sample_profile, tmp_path):
        output = tmp_path / "replica.csv"
        result = runner.invoke(
            app, ["generate", str(sample_profile), "-o", str(output), "--format", "csv"]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_generate_with_rows(self, runner, sample_profile, tmp_path):
        output = tmp_path / "replica.parquet"
        result = runner.invoke(
            app, ["generate", str(sample_profile), "-o", str(output), "--rows", "25"]
        )
        assert result.exit_code == 0
        df = pl.read_parquet(output)
        assert len(df) == 25

    def test_generate_validate_fail(self, runner, tmp_path):
        """Generate with --validate fail and a profile that will mismatch."""
        # Profile expects null_fraction=0.9, but generation will produce ~0
        p = Profile(
            version="0.2.0",
            seed=42,
            row_count=50,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.9},
                ),
            ],
        )
        profile_path = tmp_path / "p.json"
        save_profile(p, profile_path)
        output = tmp_path / "r.parquet"
        result = runner.invoke(
            app,
            ["generate", str(profile_path), "-o", str(output), "--validate", "fail"],
        )
        assert result.exit_code == 1


class TestValidateCommand:
    def test_basic_validate(self, runner, sample_profile, tmp_path):
        # Generate a replica first
        replica = tmp_path / "replica.parquet"
        runner.invoke(app, ["generate", str(sample_profile), "-o", str(replica)])

        result = runner.invoke(app, ["validate", str(replica), str(sample_profile)])
        assert result.exit_code == 0

    def test_validate_json_format(self, runner, sample_profile, tmp_path):
        replica = tmp_path / "replica.parquet"
        runner.invoke(app, ["generate", str(sample_profile), "-o", str(replica)])

        result = runner.invoke(
            app, ["validate", str(replica), str(sample_profile), "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "passed" in data
        assert "checks" in data
