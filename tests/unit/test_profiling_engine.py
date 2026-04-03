"""Tests for the profiling engine."""

import datetime

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import (
    BooleanStats,
    CategoricalStats,
    DateStats,
    NumericStats,
    StringStats,
)
from proof_of_replica.core.enums import ColumnDtype, ColumnRole
from proof_of_replica.core.schema import Profile
from proof_of_replica.engines.profiling import profile_dataframe


class TestProfileDataframe:
    def test_mixed_types(self):
        df = pl.DataFrame(
            {
                "age": [25.0, 30.0, 35.0, 40.0, 45.0],
                "group": ["A", "B", "A", "B", "A"],
                "active": [True, False, True, False, True],
                "date": [
                    datetime.date(2023, 1, 1),
                    datetime.date(2023, 6, 1),
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 6, 1),
                    datetime.date(2025, 1, 1),
                ],
                "notes": ["hello", "world", "foo", "bar", "baz"],
            }
        )
        profile = profile_dataframe(df)

        assert isinstance(profile, Profile)
        assert len(profile.columns) == 5
        assert profile.row_count == 5
        assert profile.version == "0.2.0"
        assert profile.source_hash is not None

    def test_numeric_column_stats(self):
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(50, 10, size=500).tolist()})
        profile = profile_dataframe(df)

        col = profile.columns[0]
        assert col.dtype == ColumnDtype.FLOAT64
        assert isinstance(col.stats, NumericStats)
        assert col.stats.mean is not None
        assert abs(col.stats.mean - 50) < 3
        assert col.distribution is not None

    def test_categorical_column_stats(self):
        df = pl.DataFrame({"color": ["red", "blue", "green"] * 30})
        profile = profile_dataframe(df)

        col = profile.columns[0]
        assert col.dtype == ColumnDtype.CATEGORICAL
        assert isinstance(col.stats, CategoricalStats)
        assert col.stats.cardinality == 3
        assert len(col.stats.value_counts) == 3

    def test_boolean_column_stats(self):
        df = pl.DataFrame({"flag": [True, False, True, True, False] * 20})
        profile = profile_dataframe(df)

        col = profile.columns[0]
        assert col.dtype == ColumnDtype.BOOLEAN
        assert isinstance(col.stats, BooleanStats)
        assert abs(col.stats.true_fraction - 0.6) < 0.05

    def test_date_column_stats(self):
        dates = [
            datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)
        ]
        df = pl.DataFrame({"d": dates})
        profile = profile_dataframe(df)

        col = profile.columns[0]
        assert col.dtype == ColumnDtype.DATE
        assert isinstance(col.stats, DateStats)
        assert col.stats.min is not None
        assert col.stats.max is not None

    def test_string_column_stats(self):
        df = pl.DataFrame({"text": [f"unique_text_{i}" for i in range(100)]})
        profile = profile_dataframe(df)

        col = profile.columns[0]
        assert col.dtype == ColumnDtype.STRING
        assert isinstance(col.stats, StringStats)

    def test_identifier_role(self):
        df = pl.DataFrame({"id": [f"ID_{i}" for i in range(50)]})
        profile = profile_dataframe(df)

        col = profile.columns[0]
        assert col.role == ColumnRole.IDENTIFIER

    def test_with_correlations(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=200)
        y = x * 0.8 + rng.normal(0, 0.5, size=200)
        df = pl.DataFrame({"x": x.tolist(), "y": y.tolist()})
        profile = profile_dataframe(df, correlations=True)

        assert profile.correlations is not None
        assert "x" in profile.correlations.columns
        assert "y" in profile.correlations.columns

    def test_without_correlations(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        profile = profile_dataframe(df, correlations=False)
        assert profile.correlations is None

    def test_group_column(self):
        rng = np.random.default_rng(42)
        n = 200
        groups = ["A"] * (n // 2) + ["B"] * (n // 2)
        values = rng.normal(0, 1, size=n)
        # Shift group B by 10
        values[n // 2 :] += 10.0
        df = pl.DataFrame({"group": groups, "value": values.tolist()})
        profile = profile_dataframe(df, group_column="group")

        group_col = next(c for c in profile.columns if c.name == "group")
        assert group_col.role == ColumnRole.GROUP

        value_col = next(c for c in profile.columns if c.name == "value")
        assert value_col.group_effects is not None
        assert "A" in value_col.group_effects.effects
        assert "B" in value_col.group_effects.effects

    def test_profile_round_trip_json(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "label": ["a", "b", "a", "b", "a"],
            }
        )
        profile = profile_dataframe(df)
        data = profile.model_dump(mode="json")
        restored = Profile.model_validate(data)
        assert restored.version == profile.version
        assert len(restored.columns) == len(profile.columns)

    def test_seed_stored(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        profile = profile_dataframe(df, seed=99)
        assert profile.seed == 99
