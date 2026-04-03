"""Tests for differential privacy mechanisms and DP profiling."""

import json

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.engines._dp_noise import (
    allocate_budget,
    dp_histogram,
    gaussian_mechanism,
    laplace_mechanism,
)
from proof_of_replica.engines.profiling import profile_dataframe


class TestLaplaceMechanism:
    def test_adds_noise(self):
        rng = np.random.default_rng(42)
        noised = laplace_mechanism(100.0, 1.0, 1.0, rng)
        assert noised != 100.0

    def test_deterministic_with_same_seed(self):
        r1 = laplace_mechanism(50.0, 1.0, 0.5, np.random.default_rng(7))
        r2 = laplace_mechanism(50.0, 1.0, 0.5, np.random.default_rng(7))
        assert r1 == r2

    def test_higher_epsilon_less_noise(self):
        rng = np.random.default_rng(42)
        samples_low_eps = [laplace_mechanism(0.0, 1.0, 0.1, rng) for _ in range(1000)]
        rng2 = np.random.default_rng(42)
        samples_high_eps = [
            laplace_mechanism(0.0, 1.0, 10.0, rng2) for _ in range(1000)
        ]
        assert np.std(samples_low_eps) > np.std(samples_high_eps)


class TestGaussianMechanism:
    def test_adds_noise(self):
        rng = np.random.default_rng(42)
        noised = gaussian_mechanism(100.0, 1.0, 1.0, 1e-5, rng)
        assert noised != 100.0


class TestDPHistogram:
    def test_noises_counts(self):
        rng = np.random.default_rng(42)
        counts = np.array([100, 200, 300])
        noised = dp_histogram(counts, 1.0, rng)
        assert len(noised) == 3
        assert not np.array_equal(counts, noised)
        assert all(c >= 0 for c in noised)


class TestAllocateBudget:
    def test_splits_budget(self):
        per_eps, per_delta = allocate_budget(10, 1.0, 1e-5)
        assert per_eps == pytest.approx(0.1)
        assert per_delta == pytest.approx(1e-6)


class TestDPProfiling:
    def test_dp_profile_has_privacy_block(self):
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(50, 10, 200).tolist()})
        profile = profile_dataframe(df, dp=True, epsilon=1.0, delta=1e-5)
        assert profile.privacy is not None
        assert profile.privacy["mechanism"] == "differential_privacy"
        assert profile.privacy["epsilon"] == 1.0

    def test_dp_changes_stats(self):
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(50, 10, 200).tolist()})

        profile_no_dp = profile_dataframe(df, dp=False, seed=42)
        profile_dp = profile_dataframe(df, dp=True, epsilon=0.5, delta=1e-5, seed=42)

        # Stats should differ due to DP noise
        stats_no_dp = profile_no_dp.columns[0].stats
        stats_dp = profile_dp.columns[0].stats
        assert stats_no_dp is not None
        assert stats_dp is not None
        # With low epsilon, noise is significant
        assert stats_no_dp.mean != stats_dp.mean

    def test_dp_without_dp_flag_no_privacy(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = profile_dataframe(df, dp=False)
        assert profile.privacy is None

    def test_dp_categorical_column(self):
        df = pl.DataFrame({"c": ["a", "b", "c"] * 30})
        profile = profile_dataframe(df, dp=True, epsilon=1.0)
        col = profile.columns[0]
        assert col.stats is not None

    def test_dp_boolean_column(self):
        df = pl.DataFrame({"b": [True, False] * 50})
        profile = profile_dataframe(df, dp=True, epsilon=1.0)
        col = profile.columns[0]
        assert col.stats is not None


class TestDPCLI:
    def test_profile_with_dp(self, tmp_path):
        runner = CliRunner()
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist()})
        csv_path = tmp_path / "data.csv"
        df.write_csv(csv_path)

        output = tmp_path / "dp_profile.json"
        result = runner.invoke(
            app,
            [
                "profile",
                str(csv_path),
                "-o",
                str(output),
                "--dp",
                "--epsilon",
                "1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

        data = json.loads(output.read_text())
        assert data.get("privacy") is not None
        assert data["privacy"]["mechanism"] == "differential_privacy"
