"""Integration tests for the full proof-of-replica pipeline."""

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
def sample_data(tmp_path: Path) -> Path:
    """Create a realistic sample CSV with multiple column types."""
    rng = np.random.default_rng(42)
    n = 200

    df = pl.DataFrame(
        {
            "age": rng.normal(50, 10, n).tolist(),
            "score": rng.uniform(0, 100, n).tolist(),
            "group": (["treatment"] * (n // 2) + ["control"] * (n // 2)),
            "active": ([True, False] * (n // 2)),
        }
    )
    path = tmp_path / "data.csv"
    df.write_csv(path)
    return path


@pytest.mark.integration
class TestProfileGenerateValidateReport:
    """End-to-end: profile CSV → generate replica → validate → report."""

    def test_full_pipeline(self, runner, sample_data, tmp_path):
        profile_path = tmp_path / "profile.json"
        replica_path = tmp_path / "replica.parquet"
        report_path = tmp_path / "report.html"

        # Step 1: Profile
        result = runner.invoke(
            app,
            ["profile", str(sample_data), "-o", str(profile_path), "--correlations"],
        )
        assert result.exit_code == 0, result.output
        assert profile_path.exists()

        profile_data = json.loads(profile_path.read_text())
        assert len(profile_data["columns"]) == 4
        assert profile_data["correlations"] is not None

        # Step 2: Generate
        result = runner.invoke(
            app,
            [
                "generate",
                str(profile_path),
                "-o",
                str(replica_path),
                "--validate",
                "warn",
            ],
        )
        assert result.exit_code == 0, result.output
        assert replica_path.exists()
        replica = pl.read_parquet(replica_path)
        assert len(replica) == 200
        assert replica.columns == ["age", "score", "group", "active"]

        # Step 3: Validate
        result = runner.invoke(
            app, ["validate", str(replica_path), str(profile_path), "--format", "json"]
        )
        assert result.exit_code == 0
        validation = json.loads(result.output)
        assert "passed" in validation

        # Step 4: Report
        result = runner.invoke(
            app,
            [
                "report",
                str(replica_path),
                str(profile_path),
                "-o",
                str(report_path),
                "--original",
                str(sample_data),
            ],
        )
        assert result.exit_code == 0
        assert report_path.exists()
        html = report_path.read_text()
        assert "Fidelity Report" in html
        assert "data:image/png;base64," in html


@pytest.mark.integration
class TestDiffPipeline:
    """End-to-end: profile → generate → diff original vs replica."""

    def test_diff_with_privacy(self, runner, sample_data, tmp_path):
        profile_path = tmp_path / "profile.json"
        replica_path = tmp_path / "replica.csv"

        runner.invoke(app, ["profile", str(sample_data), "-o", str(profile_path)])
        runner.invoke(
            app,
            ["generate", str(profile_path), "-o", str(replica_path), "--format", "csv"],
        )

        result = runner.invoke(
            app, ["diff", str(sample_data), str(replica_path), "--privacy"]
        )
        assert result.exit_code == 0
        assert "Privacy distance" in result.output
        assert "Column: age" in result.output


@pytest.mark.integration
class TestGenerateWithOverrides:
    """End-to-end: profile → generate with overrides."""

    def test_override_seed(self, runner, sample_data, tmp_path):
        profile_path = tmp_path / "profile.json"
        replica1 = tmp_path / "r1.parquet"
        replica2 = tmp_path / "r2.parquet"

        runner.invoke(app, ["profile", str(sample_data), "-o", str(profile_path)])

        runner.invoke(
            app,
            [
                "generate",
                str(profile_path),
                "-o",
                str(replica1),
                "--override",
                '{"seed": 1}',
            ],
        )
        runner.invoke(
            app,
            [
                "generate",
                str(profile_path),
                "-o",
                str(replica2),
                "--override",
                '{"seed": 2}',
            ],
        )

        df1 = pl.read_parquet(replica1)
        df2 = pl.read_parquet(replica2)
        assert df1["age"].to_list() != df2["age"].to_list()


@pytest.mark.integration
class TestReplicateThenReport:
    """End-to-end: replicate → report."""

    def test_replicate_and_report(self, runner, sample_data, tmp_path):
        replica_path = tmp_path / "replica.parquet"
        profile_path = tmp_path / "saved.json"
        report_path = tmp_path / "report.html"

        result = runner.invoke(
            app,
            [
                "replicate",
                str(sample_data),
                "-o",
                str(replica_path),
                "--save-profile",
                str(profile_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert replica_path.exists()
        assert profile_path.exists()

        result = runner.invoke(
            app,
            ["report", str(replica_path), str(profile_path), "-o", str(report_path)],
        )
        assert result.exit_code == 0
        assert report_path.exists()
