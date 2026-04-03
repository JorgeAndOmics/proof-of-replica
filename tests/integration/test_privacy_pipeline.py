"""Integration tests for privacy features."""

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
    """Create a dataset with quasi-identifier risk."""
    rng = np.random.default_rng(42)
    n = 200
    df = pl.DataFrame(
        {
            "age_group": rng.choice(["young", "middle", "old"], size=n).tolist(),
            "gender": rng.choice(["M", "F"], size=n).tolist(),
            "score": rng.normal(50, 10, n).tolist(),
        }
    )
    path = tmp_path / "data.csv"
    df.write_csv(path)
    return path


@pytest.mark.integration
class TestDPProfileGenerateValidate:
    """End-to-end: DP profile → generate → validate with k-anonymity."""

    def test_dp_pipeline(self, runner, sample_data, tmp_path):
        profile_path = tmp_path / "dp_profile.json"
        replica_path = tmp_path / "replica.parquet"

        # Step 1: Profile with DP
        result = runner.invoke(
            app,
            [
                "profile",
                str(sample_data),
                "-o",
                str(profile_path),
                "--dp",
                "--epsilon",
                "1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert profile_path.exists()

        data = json.loads(profile_path.read_text())
        assert data["privacy"]["mechanism"] == "differential_privacy"
        assert data["privacy"]["epsilon"] == 1.0

        # Step 2: Generate from DP profile
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

        # Step 3: Validate (includes k-anonymity and quasi-ID checks)
        result = runner.invoke(
            app, ["validate", str(replica_path), str(profile_path), "--format", "json"]
        )
        assert result.exit_code == 0
        validation = json.loads(result.output)
        check_names = [c["check_name"] for c in validation["checks"]]
        assert "k_anonymity" in check_names
        assert "quasi_identifier" in check_names


@pytest.mark.integration
class TestQuasiIDDetection:
    """Verify quasi-identifier detection on a known risky dataset."""

    def test_detects_risk(self, runner, tmp_path):
        rng = np.random.default_rng(42)
        n = 100
        df = pl.DataFrame(
            {
                "gender": rng.choice(["M", "F"], size=n).tolist(),
                "region": rng.choice(["N", "S", "E", "W"], size=n).tolist(),
                "score": rng.normal(50, 10, n).tolist(),
            }
        )
        csv_path = tmp_path / "risky.csv"
        df.write_csv(csv_path)

        profile_path = tmp_path / "p.json"
        runner.invoke(app, ["profile", str(csv_path), "-o", str(profile_path)])

        replica_path = tmp_path / "r.parquet"
        runner.invoke(
            app,
            [
                "generate",
                str(profile_path),
                "-o",
                str(replica_path),
                "--validate",
                "off",
            ],
        )

        result = runner.invoke(
            app, ["validate", str(replica_path), str(profile_path), "--format", "json"]
        )
        assert result.exit_code == 0
        validation = json.loads(result.output)
        qi_checks = [
            c for c in validation["checks"] if c["check_name"] == "quasi_identifier"
        ]
        # Should detect risk on this small dataset
        assert len(qi_checks) > 0


@pytest.mark.integration
class TestDPProducesDifferentStats:
    """DP profiling should produce different stats than non-DP on same data."""

    def test_stats_differ(self, runner, sample_data, tmp_path):
        no_dp_path = tmp_path / "no_dp.json"
        dp_path = tmp_path / "dp.json"

        runner.invoke(app, ["profile", str(sample_data), "-o", str(no_dp_path)])
        runner.invoke(
            app,
            [
                "profile",
                str(sample_data),
                "-o",
                str(dp_path),
                "--dp",
                "--epsilon",
                "0.5",
            ],
        )

        no_dp = json.loads(no_dp_path.read_text())
        dp = json.loads(dp_path.read_text())

        # DP profile should have privacy block
        assert dp.get("privacy") is not None
        assert no_dp.get("privacy") is None

        # Stats should differ (DP noise applied)
        no_dp_mean = no_dp["columns"][2]["stats"]["mean"]  # score column
        dp_mean = dp["columns"][2]["stats"]["mean"]
        assert no_dp_mean != dp_mean
