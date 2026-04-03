"""Tests for type inference engine."""

import datetime

import polars as pl
import pytest

from proof_of_replica.core.enums import ColumnDtype
from proof_of_replica.core.schema import ProfilerConfig
from proof_of_replica.engines._type_inference import infer_dtype


@pytest.fixture
def config() -> ProfilerConfig:
    return ProfilerConfig()


class TestBooleanDetection:
    def test_native_boolean(self, config):
        s = pl.Series("x", [True, False, True, False])
        assert infer_dtype(s, 4, config) == ColumnDtype.BOOLEAN

    def test_zero_one_int(self, config):
        s = pl.Series("x", [0, 1, 0, 1, 1])
        assert infer_dtype(s, 5, config) == ColumnDtype.BOOLEAN

    def test_true_false_string(self, config):
        s = pl.Series("x", ["true", "false", "true"])
        assert infer_dtype(s, 3, config) == ColumnDtype.BOOLEAN

    def test_yes_no_string(self, config):
        s = pl.Series("x", ["yes", "no", "yes", "no"])
        assert infer_dtype(s, 4, config) == ColumnDtype.BOOLEAN

    def test_not_boolean_three_values(self, config):
        s = pl.Series("x", [0, 1, 2])
        assert infer_dtype(s, 3, config) != ColumnDtype.BOOLEAN


class TestIntegerDetection:
    def test_native_int(self, config):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        assert infer_dtype(s, 5, config) == ColumnDtype.INT64

    def test_string_integers(self, config):
        s = pl.Series("x", ["10", "20", "30", "40"])
        assert infer_dtype(s, 4, config) == ColumnDtype.INT64


class TestFloatDetection:
    def test_native_float(self, config):
        s = pl.Series("x", [1.5, 2.3, 3.7])
        assert infer_dtype(s, 3, config) == ColumnDtype.FLOAT64

    def test_string_floats(self, config):
        s = pl.Series("x", ["1.5", "2.3", "3.7", "4.1"])
        assert infer_dtype(s, 4, config) == ColumnDtype.FLOAT64


class TestDateDetection:
    def test_native_date(self, config):
        s = pl.Series("x", [datetime.date(2023, 1, 1), datetime.date(2023, 6, 15)])
        assert infer_dtype(s, 2, config) == ColumnDtype.DATE

    def test_iso_date_strings(self, config):
        s = pl.Series("x", ["2023-01-01", "2023-06-15", "2024-12-31"])
        assert infer_dtype(s, 3, config) == ColumnDtype.DATE


class TestCategoricalDetection:
    def test_low_cardinality_string(self, config):
        s = pl.Series("x", ["red", "blue", "green", "red", "blue"] * 20)
        assert infer_dtype(s, 100, config) == ColumnDtype.CATEGORICAL

    def test_high_cardinality_not_categorical(self, config):
        s = pl.Series("x", [f"item_{i}" for i in range(100)])
        assert infer_dtype(s, 100, config) != ColumnDtype.CATEGORICAL


class TestStringDetection:
    def test_high_cardinality_string(self, config):
        # All unique but not "identifier" because we test with free-form text
        values = [f"some long text {i} with variation" for i in range(50)]
        s = pl.Series("x", values)
        # n_unique == n_rows so it won't be categorical, falls through to identifier check
        # _is_identifier returns STRING (string with identifier role)
        result = infer_dtype(s, 50, config)
        assert result == ColumnDtype.STRING


class TestPriorityCascade:
    def test_zero_one_is_boolean_not_integer(self, config):
        """[0, 1] with only 2 values should be boolean, not integer."""
        s = pl.Series("x", [0, 1, 0, 1, 0])
        assert infer_dtype(s, 5, config) == ColumnDtype.BOOLEAN

    def test_float_not_date(self, config):
        """Numeric float should not be classified as date."""
        s = pl.Series("x", [1.5, 2.5, 3.5])
        assert infer_dtype(s, 3, config) == ColumnDtype.FLOAT64

    def test_empty_series_is_string(self, config):
        s = pl.Series("x", [], dtype=pl.Utf8)
        assert infer_dtype(s, 0, config) == ColumnDtype.STRING
