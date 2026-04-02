"""Tests for noise perturbation utilities."""

import datetime

import numpy as np
import polars as pl
import pytest

from proof_of_replica.core.enums import ColumnDtype
from proof_of_replica.utils.noise import apply_noise


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestNoiseNumeric:
    def test_float_noise_changes_values(self, rng):
        series = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_noise(series, ColumnDtype.FLOAT64, 0.1, rng)
        assert result.to_list() != series.to_list()
        assert result.dtype == pl.Float64

    def test_zero_noise_no_change(self, rng):
        series = pl.Series("x", [1.0, 2.0, 3.0])
        result = apply_noise(series, ColumnDtype.FLOAT64, 0.0, rng)
        assert result.to_list() == series.to_list()

    def test_int_noise_stays_integer(self, rng):
        series = pl.Series("x", [10, 20, 30, 40, 50], dtype=pl.Int64)
        result = apply_noise(series, ColumnDtype.INT64, 0.1, rng)
        assert result.dtype == pl.Int64

    def test_deterministic(self):
        series = pl.Series("x", [1.0, 2.0, 3.0])
        r1 = apply_noise(series, ColumnDtype.FLOAT64, 0.1, np.random.default_rng(7))
        r2 = apply_noise(series, ColumnDtype.FLOAT64, 0.1, np.random.default_rng(7))
        assert r1.to_list() == r2.to_list()

    def test_preserves_nulls(self, rng):
        series = pl.Series("x", [1.0, None, 3.0], dtype=pl.Float64)
        result = apply_noise(series, ColumnDtype.FLOAT64, 0.1, rng)
        assert result[1] is None


class TestNoiseCategorical:
    def test_flips_labels(self, rng):
        values = ["a", "b"] * 500
        series = pl.Series("x", values)
        result = apply_noise(series, ColumnDtype.CATEGORICAL, 0.5, rng)
        # With noise=0.5, some labels should have flipped
        a_count = result.to_list().count("a")
        assert a_count != 500  # Should have some flips

    def test_no_flip_at_zero(self, rng):
        series = pl.Series("x", ["a", "b", "c"])
        result = apply_noise(series, ColumnDtype.CATEGORICAL, 0.0, rng)
        assert result.to_list() == ["a", "b", "c"]

    def test_single_label_no_change(self, rng):
        series = pl.Series("x", ["only", "only", "only"])
        result = apply_noise(series, ColumnDtype.CATEGORICAL, 0.5, rng)
        assert result.to_list() == ["only", "only", "only"]


class TestNoiseBoolean:
    def test_flips_bits(self, rng):
        values = [True] * 1000
        series = pl.Series("x", values, dtype=pl.Boolean)
        result = apply_noise(series, ColumnDtype.BOOLEAN, 0.3, rng)
        false_count = result.to_list().count(False)
        assert 200 < false_count < 400

    def test_preserves_nulls(self, rng):
        series = pl.Series("x", [True, None, False], dtype=pl.Boolean)
        result = apply_noise(series, ColumnDtype.BOOLEAN, 0.5, rng)
        assert result[1] is None


class TestNoiseDate:
    def test_shifts_dates(self, rng):
        dates = [datetime.date(2023, 6, 15)] * 100
        series = pl.Series("x", dates, dtype=pl.Date)
        result = apply_noise(series, ColumnDtype.DATE, 0.1, rng)
        assert result.dtype == pl.Date
        # With single-date range, range_days=1, so shifts should be small
        assert len(result) == 100


class TestNoiseEdgeCases:
    def test_empty_series(self, rng):
        series = pl.Series("x", [], dtype=pl.Float64)
        result = apply_noise(series, ColumnDtype.FLOAT64, 0.1, rng)
        assert len(result) == 0

    def test_string_no_noise(self, rng):
        series = pl.Series("x", ["hello", "world"])
        result = apply_noise(series, ColumnDtype.STRING, 0.5, rng)
        assert result.to_list() == ["hello", "world"]
