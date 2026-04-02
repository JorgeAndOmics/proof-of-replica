"""Tests for date column generator."""

import datetime

import numpy as np
import polars as pl
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.date import generate_date


class TestGenerateDate:
    def test_returns_date_series(self, rng):
        col = ColumnDefinition(
            name="d",
            dtype="date",
            stats={"min": "2020-01-01", "max": "2025-12-31"},
        )
        result = generate_date(col, 100, rng)
        assert isinstance(result, pl.Series)
        assert result.dtype == pl.Date
        assert result.name == "d"
        assert len(result) == 100

    def test_values_in_range(self, rng):
        col = ColumnDefinition(
            name="d",
            dtype="date",
            stats={"min": "2023-06-01", "max": "2023-06-30"},
        )
        result = generate_date(col, 500, rng)
        dates = result.to_list()
        for d in dates:
            assert datetime.date(2023, 6, 1) <= d <= datetime.date(2023, 6, 30)

    def test_deterministic(self):
        col = ColumnDefinition(
            name="d",
            dtype="date",
            stats={"min": "2020-01-01", "max": "2020-12-31"},
        )
        r1 = generate_date(col, 50, np.random.default_rng(11))
        r2 = generate_date(col, 50, np.random.default_rng(11))
        assert r1.to_list() == r2.to_list()

    def test_error_on_missing_stats(self, rng):
        col = ColumnDefinition(name="d", dtype="date")
        with pytest.raises(GenerationError, match="DateStats"):
            generate_date(col, 10, rng)

    def test_error_on_missing_min_max(self, rng):
        col = ColumnDefinition(name="d", dtype="date", stats={"min": "2020-01-01"})
        with pytest.raises(GenerationError, match="min and max"):
            generate_date(col, 10, rng)
