"""Tests for quasi-identifier detection and k-anonymity checks."""

import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import Profile, ValidationConfig
from proof_of_replica.engines.validation import CheckStatus, validate


def _make_profile(columns: list[ColumnDefinition], **kwargs: object) -> Profile:
    defaults = {"version": "0.2.0", "seed": 42}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Profile(columns=columns, **defaults)  # type: ignore[arg-type]


class TestKAnonymity:
    def test_passes_with_large_groups(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="group",
                    dtype="categorical",
                    stats={"value_counts": {"A": 0.5, "B": 0.5}},
                ),
            ],
        )
        df = pl.DataFrame({"group": ["A"] * 50 + ["B"] * 50})
        result = validate(df, profile)
        k_check = next(c for c in result.checks if c.check_name == "k_anonymity")
        assert k_check.status == CheckStatus.PASS

    def test_warns_with_small_groups(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="group",
                    dtype="categorical",
                    stats={"value_counts": {"A": 0.5, "B": 0.3, "C": 0.2}},
                ),
            ],
            validation=ValidationConfig(k_anonymity_k=20),
        )
        # C has only 2 rows
        df = pl.DataFrame({"group": ["A"] * 50 + ["B"] * 30 + ["C"] * 2})
        result = validate(df, profile)
        k_check = next(c for c in result.checks if c.check_name == "k_anonymity")
        assert k_check.status == CheckStatus.WARN

    def test_no_categorical_columns(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = validate(df, profile)
        k_checks = [c for c in result.checks if c.check_name == "k_anonymity"]
        assert len(k_checks) == 0


class TestQuasiIdentifiers:
    def test_detects_risky_combination(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="gender",
                    dtype="categorical",
                    stats={"value_counts": {"M": 0.5, "F": 0.5}},
                ),
                ColumnDefinition(
                    name="age_group",
                    dtype="categorical",
                    stats={"value_counts": {"young": 0.5, "old": 0.5}},
                ),
            ],
            validation=ValidationConfig(quasi_id_max_k=10),
        )
        # Each combo has only 1 row — risky
        df = pl.DataFrame(
            {
                "gender": ["M", "F", "M", "F"],
                "age_group": ["young", "young", "old", "old"],
            }
        )
        result = validate(df, profile)
        qi_checks = [c for c in result.checks if c.check_name == "quasi_identifier"]
        assert any(c.status == CheckStatus.WARN for c in qi_checks)

    def test_passes_with_large_groups(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="a",
                    dtype="categorical",
                    stats={"value_counts": {"X": 0.5, "Y": 0.5}},
                ),
                ColumnDefinition(
                    name="b",
                    dtype="categorical",
                    stats={"value_counts": {"1": 0.5, "2": 0.5}},
                ),
            ],
        )
        # Each combo has 25 rows — safe
        df = pl.DataFrame(
            {
                "a": ["X", "Y"] * 50,
                "b": ["1", "2"] * 50,
            }
        )
        result = validate(df, profile)
        qi_checks = [c for c in result.checks if c.check_name == "quasi_identifier"]
        assert all(c.status == CheckStatus.PASS for c in qi_checks)

    def test_single_categorical_no_check(self):
        """Need at least 2 categorical columns for quasi-ID check."""
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="group",
                    dtype="categorical",
                    stats={"value_counts": {"A": 1.0}},
                ),
            ],
        )
        df = pl.DataFrame({"group": ["A"] * 10})
        result = validate(df, profile)
        qi_checks = [c for c in result.checks if c.check_name == "quasi_identifier"]
        assert len(qi_checks) == 0
