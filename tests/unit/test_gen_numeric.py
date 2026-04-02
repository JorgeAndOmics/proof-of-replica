"""Tests for numeric column generator."""

import numpy as np
import polars as pl
import pytest
from scipy import stats as sp_stats

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.numeric import generate_numeric


class TestGenerateNumericFloat:
    def test_returns_float64_series(self, numeric_col, rng):
        result = generate_numeric(numeric_col, 100, rng)
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Float64
        assert result.name == "age"
        assert len(result) == 100

    def test_fallback_normal_distribution(self, numeric_col, rng):
        result = generate_numeric(numeric_col, 10_000, rng)
        values = result.to_numpy()
        assert abs(values.mean() - 50.0) < 1.0
        assert abs(values.std() - 10.0) < 1.0

    def test_explicit_normal_distribution(self, rng):
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 0.0, "std": 1.0},
            distribution={"family": "normal", "params": {"loc": 0.0, "scale": 1.0}},
        )
        result = generate_numeric(col, 5000, rng)
        values = result.to_numpy()
        _, p = sp_stats.kstest(values, "norm", args=(0.0, 1.0))
        assert p > 0.01

    def test_lognormal_distribution(self, rng):
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 10.0, "std": 5.0},
            distribution={
                "family": "lognormal",
                "params": {"s": 0.5, "loc": 0.0, "scale": 1.0},
            },
        )
        result = generate_numeric(col, 1000, rng)
        assert (result.to_numpy() >= 0).all()

    def test_uniform_distribution(self, rng):
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 5.0, "std": 2.9},
            distribution={
                "family": "uniform",
                "params": {"loc": 0.0, "scale": 10.0},
            },
        )
        result = generate_numeric(col, 5000, rng)
        values = result.to_numpy()
        assert values.min() >= 0.0
        assert values.max() <= 10.0

    def test_deterministic(self, numeric_col):
        r1 = generate_numeric(numeric_col, 50, np.random.default_rng(7))
        r2 = generate_numeric(numeric_col, 50, np.random.default_rng(7))
        assert r1.to_list() == r2.to_list()

    def test_fallback_missing_std(self, rng):
        col = ColumnDefinition(name="x", dtype="float64", stats={"mean": 100.0})
        result = generate_numeric(col, 100, rng)
        assert len(result) == 100

    def test_error_on_missing_stats(self, rng):
        col = ColumnDefinition(name="x", dtype="float64")
        with pytest.raises(GenerationError, match="NumericStats"):
            generate_numeric(col, 10, rng)

    def test_error_on_missing_mean(self, rng):
        col = ColumnDefinition(name="x", dtype="float64", stats={"std": 1.0})
        with pytest.raises(GenerationError, match="without mean"):
            generate_numeric(col, 10, rng)


class TestGenerateNumericInt:
    def test_returns_int64_series(self, int_col, rng):
        result = generate_numeric(int_col, 100, rng)
        assert result.dtype == pl.Int64
        assert len(result) == 100

    def test_values_are_integers(self, int_col, rng):
        result = generate_numeric(int_col, 100, rng)
        values = result.to_numpy()
        np.testing.assert_array_equal(values, values.astype(np.int64))


class TestEmpiricalDistributions:
    def test_empirical_kde(self, rng):
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 5.0, "std": 2.0},
            distribution={
                "family": "empirical_kde",
                "params": {"bandwidth": 2.0},
            },
        )
        result = generate_numeric(col, 100, rng)
        assert len(result) == 100
        assert result.dtype == pl.Float64

    def test_empirical_histogram(self, rng):
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 5.0, "std": 2.0},
            distribution={
                "family": "empirical_histogram",
                "params": {
                    "edges": "[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]",
                    "counts": "[10, 20, 40, 20, 10]",
                },
            },
        )
        result = generate_numeric(col, 1000, rng)
        values = result.to_numpy()
        assert values.min() >= 0.0
        assert values.max() <= 10.0

    def test_empirical_histogram_missing_params(self, rng):
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 5.0, "std": 2.0},
            distribution={"family": "empirical_histogram", "params": {}},
        )
        with pytest.raises(GenerationError, match="edges"):
            generate_numeric(col, 10, rng)
