"""Tests for missingness injection."""

import numpy as np
import polars as pl
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import MissingnessConfig, Profile
from proof_of_replica.utils.missingness import inject_missingness


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_profile(
    columns: list[ColumnDefinition],
    missingness: MissingnessConfig | None = None,
) -> Profile:
    return Profile(version="0.2.0", columns=columns, missingness=missingness)


class TestMCARInjection:
    def test_injects_nulls(self, rng):
        profile = _make_profile(
            [
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.3},
                ),
            ]
        )
        df = pl.DataFrame({"x": [1.0] * 1000})
        result = inject_missingness(df, profile, rng)
        null_frac = result["x"].null_count() / len(result)
        assert abs(null_frac - 0.3) < 0.05

    def test_no_nulls_when_fraction_zero(self, rng):
        profile = _make_profile(
            [
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.0},
                ),
            ]
        )
        df = pl.DataFrame({"x": [1.0] * 100})
        result = inject_missingness(df, profile, rng)
        assert result["x"].null_count() == 0

    def test_deterministic(self):
        profile = _make_profile(
            [
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.2},
                ),
            ]
        )
        df = pl.DataFrame({"x": [1.0] * 100})
        r1 = inject_missingness(df, profile, np.random.default_rng(7))
        r2 = inject_missingness(df, profile, np.random.default_rng(7))
        assert r1["x"].is_null().to_list() == r2["x"].is_null().to_list()

    def test_no_stats_no_nulls(self, rng):
        profile = _make_profile(
            [
                ColumnDefinition(name="x", dtype="float64"),
            ]
        )
        df = pl.DataFrame({"x": [1.0] * 100})
        result = inject_missingness(df, profile, rng)
        assert result["x"].null_count() == 0


class TestCoMissingness:
    def test_co_missing_propagates(self, rng):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="a",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.5},
                ),
                ColumnDefinition(
                    name="b",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.0},
                ),
            ],
            missingness=MissingnessConfig(
                pattern="MCAR",
                co_missing={"a,b": 0.8},
            ),
        )
        df = pl.DataFrame({"a": [1.0] * 1000, "b": [2.0] * 1000})
        result = inject_missingness(df, profile, rng)

        # b should have some nulls due to co-missingness (a has ~50% null, 80% co-miss)
        assert result["b"].is_null().sum() > 100

    def test_no_co_missing_config(self, rng):
        profile = _make_profile(
            [
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ]
        )
        df = pl.DataFrame({"x": [1.0] * 100})
        result = inject_missingness(df, profile, rng)
        assert result["x"].null_count() == 0
