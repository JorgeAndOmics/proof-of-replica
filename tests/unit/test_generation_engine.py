"""Tests for the generation pipeline orchestrator."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import CorrelationConfig, Profile
from proof_of_replica.engines.generation import generate


def _make_profile(**kwargs: object) -> Profile:
    defaults = {
        "version": "0.2.0",
        "seed": 42,
    }
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Profile(**defaults)  # type: ignore[arg-type]


class TestGenerate:
    def test_minimal_single_column(self):
        profile = _make_profile(
            row_count=100,
            columns=[
                ColumnDefinition(
                    name="age",
                    dtype="float64",
                    stats={"mean": 50.0, "std": 10.0},
                ),
            ],
        )
        df = generate(profile)
        assert df.shape == (100, 1)
        assert df.columns == ["age"]
        assert df["age"].dtype == pl.Float64

    def test_mixed_column_types(self):
        profile = _make_profile(
            row_count=50,
            columns=[
                ColumnDefinition(
                    name="id",
                    dtype="string",
                    role="identifier",
                    generator={"method": "sequential", "prefix": "S_", "zero_pad": 3},
                ),
                ColumnDefinition(
                    name="age",
                    dtype="float64",
                    stats={"mean": 50.0, "std": 10.0},
                ),
                ColumnDefinition(
                    name="group",
                    dtype="categorical",
                    role="group",
                    stats={"value_counts": {"A": 0.5, "B": 0.5}},
                ),
                ColumnDefinition(
                    name="active",
                    dtype="boolean",
                    stats={"true_fraction": 0.7},
                ),
                ColumnDefinition(
                    name="date",
                    dtype="date",
                    stats={"min": "2020-01-01", "max": "2025-12-31"},
                ),
            ],
        )
        df = generate(profile)
        assert df.shape == (50, 5)
        assert df["id"].dtype == pl.Utf8
        assert df["age"].dtype == pl.Float64
        assert df["group"].dtype == pl.Utf8
        assert df["active"].dtype == pl.Boolean
        assert df["date"].dtype == pl.Date

    def test_deterministic(self):
        profile = _make_profile(
            row_count=50,
            seed=123,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df1 = generate(profile)
        df2 = generate(profile)
        assert df1["x"].to_list() == df2["x"].to_list()

    def test_seed_override(self):
        profile = _make_profile(
            row_count=50,
            seed=1,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df1 = generate(profile, seed=99)
        df2 = generate(profile, seed=99)
        df3 = generate(profile, seed=1)
        assert df1["x"].to_list() == df2["x"].to_list()
        assert df1["x"].to_list() != df3["x"].to_list()

    def test_row_count_override(self):
        profile = _make_profile(
            row_count=100,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df = generate(profile, row_count=25)
        assert len(df) == 25

    def test_default_row_count(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df = generate(profile)
        assert len(df) == 1000


class TestGroupEffects:
    def test_group_shift_applied(self):
        profile = _make_profile(
            row_count=2000,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="group",
                    dtype="categorical",
                    role="group",
                    stats={"value_counts": {"A": 0.5, "B": 0.5}},
                ),
                ColumnDefinition(
                    name="value",
                    dtype="float64",
                    stats={"mean": 0.0, "std": 1.0},
                    group_effects={
                        "group_column": "group",
                        "effects": {
                            "A": {"shift": 0.0, "scale_factor": 1.0},
                            "B": {"shift": 10.0, "scale_factor": 1.0},
                        },
                    },
                ),
            ],
        )
        df = generate(profile)
        group_b = df.filter(pl.col("group") == "B")
        group_a = df.filter(pl.col("group") == "A")
        # Group B should be shifted ~10 units higher than A
        assert group_b["value"].mean() - group_a["value"].mean() > 8.0


class TestConstraints:
    def test_clamp_applied(self):
        profile = _make_profile(
            row_count=1000,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 50.0, "std": 20.0},
                    constraints={"min": 20.0, "max": 80.0},
                ),
            ],
        )
        df = generate(profile)
        values = df["x"].to_numpy()
        assert values.min() >= 20.0
        assert values.max() <= 80.0


class TestCorrelation:
    def test_correlation_induced(self):
        profile = _make_profile(
            row_count=2000,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="a", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="b", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            correlations=CorrelationConfig(
                columns=["a", "b"],
                matrix=[[1.0, 0.7], [0.7, 1.0]],
            ),
        )
        df = generate(profile)
        corr = np.corrcoef(df["a"].to_numpy(), df["b"].to_numpy())[0, 1]
        assert abs(corr - 0.7) < 0.15


class TestMissingness:
    def test_nulls_injected(self):
        profile = _make_profile(
            row_count=1000,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.2},
                ),
            ],
        )
        df = generate(profile)
        null_frac = df["x"].null_count() / len(df)
        assert abs(null_frac - 0.2) < 0.05


class TestPrivacyCheck:
    def test_privacy_check_runs(self):
        profile = _make_profile(
            row_count=50,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        original = pl.DataFrame({"x": list(range(50))})
        df = generate(profile, original_data=original)
        assert len(df) == 50
