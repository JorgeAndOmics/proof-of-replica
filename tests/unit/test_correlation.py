"""Tests for correlation induction."""

import numpy as np
import polars as pl
import pytest

from proof_of_replica.utils.correlation import induce_correlation


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestInduceCorrelation:
    def test_identity_matrix_preserves_data(self, rng):
        n = 1000
        df = pl.DataFrame(
            {
                "a": np.random.default_rng(1).normal(0, 1, n),
                "b": np.random.default_rng(2).normal(0, 1, n),
            }
        )
        target = [[1.0, 0.0], [0.0, 1.0]]
        result = induce_correlation(df, ["a", "b"], target, rng)
        # With identity target, correlation should be near 0
        corr = np.corrcoef(result["a"].to_numpy(), result["b"].to_numpy())[0, 1]
        assert abs(corr) < 0.15

    def test_strong_positive_correlation(self, rng):
        n = 2000
        df = pl.DataFrame(
            {
                "a": np.random.default_rng(1).normal(50, 10, n),
                "b": np.random.default_rng(2).normal(100, 20, n),
            }
        )
        target = [[1.0, 0.8], [0.8, 1.0]]
        result = induce_correlation(df, ["a", "b"], target, rng)
        corr = np.corrcoef(result["a"].to_numpy(), result["b"].to_numpy())[0, 1]
        assert abs(corr - 0.8) < 0.1

    def test_negative_correlation(self, rng):
        n = 2000
        df = pl.DataFrame(
            {
                "a": np.random.default_rng(1).normal(0, 1, n),
                "b": np.random.default_rng(2).normal(0, 1, n),
            }
        )
        target = [[1.0, -0.6], [-0.6, 1.0]]
        result = induce_correlation(df, ["a", "b"], target, rng)
        corr = np.corrcoef(result["a"].to_numpy(), result["b"].to_numpy())[0, 1]
        assert abs(corr - (-0.6)) < 0.15

    def test_preserves_marginal_distribution(self, rng):
        n = 1000
        original_a = np.random.default_rng(1).normal(50, 10, n)
        df = pl.DataFrame(
            {
                "a": original_a,
                "b": np.random.default_rng(2).normal(0, 1, n),
            }
        )
        target = [[1.0, 0.5], [0.5, 1.0]]
        result = induce_correlation(df, ["a", "b"], target, rng)
        # Marginal values should be a permutation of original sorted values
        assert abs(result["a"].mean() - np.mean(original_a)) < 1.0  # type: ignore[arg-type]

    def test_single_column_noop(self, rng):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = induce_correlation(df, ["a"], [[1.0]], rng)
        assert result["a"].to_list() == df["a"].to_list()

    def test_three_columns(self, rng):
        n = 2000
        df = pl.DataFrame(
            {
                "a": np.random.default_rng(1).normal(0, 1, n),
                "b": np.random.default_rng(2).normal(0, 1, n),
                "c": np.random.default_rng(3).normal(0, 1, n),
            }
        )
        target = [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
        result = induce_correlation(df, ["a", "b", "c"], target, rng)
        corr_ab = np.corrcoef(result["a"].to_numpy(), result["b"].to_numpy())[0, 1]
        assert abs(corr_ab - 0.5) < 0.15

    def test_other_columns_unchanged(self, rng):
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "extra": ["x", "y", "z"],
            }
        )
        target = [[1.0, 0.5], [0.5, 1.0]]
        result = induce_correlation(df, ["a", "b"], target, rng)
        assert result["extra"].to_list() == ["x", "y", "z"]
