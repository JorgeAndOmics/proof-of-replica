"""Tests for boolean column generator."""

import numpy as np
import polars as pl
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.boolean import generate_boolean


class TestGenerateBoolean:
    def test_returns_boolean_series(self, boolean_col, rng):
        result = generate_boolean(boolean_col, 100, rng)
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Boolean
        assert result.name == "is_control"
        assert len(result) == 100

    def test_distribution_shape(self, boolean_col, rng):
        result = generate_boolean(boolean_col, 10_000, rng)
        true_frac = result.sum() / len(result)
        assert abs(true_frac - 0.4) < 0.03

    def test_deterministic(self, boolean_col):
        r1 = generate_boolean(boolean_col, 50, np.random.default_rng(99))
        r2 = generate_boolean(boolean_col, 50, np.random.default_rng(99))
        assert r1.to_list() == r2.to_list()

    def test_all_true(self, rng):
        col = ColumnDefinition(
            name="flag", dtype="boolean", stats={"true_fraction": 1.0}
        )
        result = generate_boolean(col, 100, rng)
        assert result.all()

    def test_all_false(self, rng):
        col = ColumnDefinition(
            name="flag", dtype="boolean", stats={"true_fraction": 0.0}
        )
        result = generate_boolean(col, 100, rng)
        assert not result.any()

    def test_error_on_wrong_stats(self, rng):
        col = ColumnDefinition(name="x", dtype="boolean")
        with pytest.raises(GenerationError, match="BooleanStats"):
            generate_boolean(col, 10, rng)
