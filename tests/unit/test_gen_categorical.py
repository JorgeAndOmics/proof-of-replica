"""Tests for categorical column generator."""

import numpy as np
import polars as pl
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.categorical import generate_categorical


class TestGenerateCategorical:
    def test_returns_utf8_series(self, categorical_col, rng):
        result = generate_categorical(categorical_col, 100, rng)
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Utf8
        assert result.name == "color"
        assert len(result) == 100

    def test_only_expected_labels(self, categorical_col, rng):
        result = generate_categorical(categorical_col, 1000, rng)
        unique = set(result.to_list())
        assert unique <= {"red", "green", "blue"}

    def test_distribution_shape(self, categorical_col, rng):
        result = generate_categorical(categorical_col, 10_000, rng)
        counts = result.value_counts()
        total = len(result)
        for row in counts.iter_rows(named=True):
            label = row["color"]
            frac = row["count"] / total
            expected = {"red": 0.5, "green": 0.3, "blue": 0.2}[label]
            assert abs(frac - expected) < 0.03

    def test_deterministic(self, categorical_col):
        r1 = generate_categorical(categorical_col, 50, np.random.default_rng(7))
        r2 = generate_categorical(categorical_col, 50, np.random.default_rng(7))
        assert r1.to_list() == r2.to_list()

    def test_unnormalized_weights(self, rng):
        col = ColumnDefinition(
            name="status",
            dtype="categorical",
            stats={"value_counts": {"on": 3.0, "off": 7.0}},
        )
        result = generate_categorical(col, 10_000, rng)
        on_frac = result.to_list().count("on") / len(result)
        assert abs(on_frac - 0.3) < 0.03

    def test_error_on_wrong_stats(self, rng):
        col = ColumnDefinition(name="x", dtype="categorical")
        with pytest.raises(GenerationError, match="CategoricalStats"):
            generate_categorical(col, 10, rng)

    def test_error_on_empty_value_counts(self, rng):
        col = ColumnDefinition(
            name="x", dtype="categorical", stats={"value_counts": {}}
        )
        with pytest.raises(GenerationError, match="empty"):
            generate_categorical(col, 10, rng)
