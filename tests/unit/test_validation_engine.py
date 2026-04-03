"""Tests for the validation engine."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import CorrelationConfig, Profile
from proof_of_replica.engines.validation import CheckStatus, ValidationResult, validate


def _make_profile(**kwargs: object) -> Profile:
    defaults = {"version": "0.2.0", "seed": 42}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Profile(**defaults)  # type: ignore[arg-type]


class TestValidateColumnNames:
    def test_matching_columns_pass(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ]
        )
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = validate(df, profile)
        name_check = next(c for c in result.checks if c.check_name == "column_names")
        assert name_check.status == CheckStatus.PASS

    def test_mismatched_columns_fail(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ]
        )
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0]})
        result = validate(df, profile)
        name_check = next(c for c in result.checks if c.check_name == "column_names")
        assert name_check.status == CheckStatus.FAIL


class TestValidateNullFraction:
    def test_correct_null_fraction_passes(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.0},
                )
            ]
        )
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = validate(df, profile)
        null_check = next(c for c in result.checks if c.check_name == "null_fraction")
        assert null_check.status == CheckStatus.PASS

    def test_wrong_null_fraction_fails(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.0},
                )
            ]
        )
        df = pl.DataFrame({"x": [1.0, None, None, None, None]})
        result = validate(df, profile)
        null_check = next(c for c in result.checks if c.check_name == "null_fraction")
        assert null_check.status == CheckStatus.FAIL


class TestValidateNumeric:
    def test_range_check_passes(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 50, "std": 10, "min": 20.0, "max": 80.0},
                )
            ]
        )
        df = pl.DataFrame({"x": [22.0, 50.0, 78.0]})
        result = validate(df, profile)
        range_check = next(
            (c for c in result.checks if c.check_name == "value_range"), None
        )
        assert range_check is not None
        assert range_check.status == CheckStatus.PASS

    def test_range_check_fails(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 50, "std": 10, "min": 20.0, "max": 80.0},
                )
            ]
        )
        df = pl.DataFrame({"x": [0.0, 50.0, 200.0]})
        result = validate(df, profile)
        range_check = next(
            (c for c in result.checks if c.check_name == "value_range"), None
        )
        assert range_check is not None
        assert range_check.status == CheckStatus.FAIL

    def test_ks_test_runs(self):
        rng = np.random.default_rng(42)
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1},
                    distribution={
                        "family": "normal",
                        "params": {"loc": 0.0, "scale": 1.0},
                    },
                )
            ]
        )
        df = pl.DataFrame({"x": rng.normal(0, 1, size=500).tolist()})
        result = validate(df, profile)
        ks_check = next((c for c in result.checks if c.check_name == "ks_test"), None)
        assert ks_check is not None


class TestValidateCategorical:
    def test_correct_cardinality_passes(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={
                        "cardinality": 3,
                        "value_counts": {"a": 0.33, "b": 0.33, "c": 0.34},
                    },
                )
            ]
        )
        df = pl.DataFrame({"c": ["a", "b", "c"] * 20})
        result = validate(df, profile)
        card_check = next(c for c in result.checks if c.check_name == "cardinality")
        assert card_check.status == CheckStatus.PASS

    def test_wrong_cardinality_fails(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={"cardinality": 3, "value_counts": {"a": 0.5, "b": 0.5}},
                )
            ]
        )
        df = pl.DataFrame({"c": ["a", "b"] * 20})
        result = validate(df, profile)
        card_check = next(c for c in result.checks if c.check_name == "cardinality")
        assert card_check.status == CheckStatus.FAIL

    def test_chisq_runs(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={"cardinality": 2, "value_counts": {"a": 0.5, "b": 0.5}},
                )
            ]
        )
        df = pl.DataFrame({"c": ["a", "b"] * 50})
        result = validate(df, profile)
        chisq = next((c for c in result.checks if c.check_name == "chisq_test"), None)
        assert chisq is not None


class TestValidateBoolean:
    def test_correct_fraction_passes(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="b", dtype="boolean", stats={"true_fraction": 0.5}
                )
            ]
        )
        df = pl.DataFrame({"b": [True, False] * 50})
        result = validate(df, profile)
        bool_check = next(
            c for c in result.checks if c.check_name == "boolean_fraction"
        )
        assert bool_check.status == CheckStatus.PASS

    def test_wrong_fraction_fails(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="b", dtype="boolean", stats={"true_fraction": 0.9}
                )
            ]
        )
        df = pl.DataFrame({"b": [True, False] * 50})
        result = validate(df, profile)
        bool_check = next(
            c for c in result.checks if c.check_name == "boolean_fraction"
        )
        assert bool_check.status == CheckStatus.FAIL


class TestValidateCorrelation:
    def test_matching_correlation_passes(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = x * 0.8 + rng.normal(0, 0.5, 500)
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="y", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            correlations=CorrelationConfig(
                columns=["x", "y"],
                matrix=np.corrcoef(x, y).tolist(),
            ),
        )
        df = pl.DataFrame({"x": x.tolist(), "y": y.tolist()})
        result = validate(df, profile)
        corr_check = next(
            c for c in result.checks if c.check_name == "correlation_frobenius"
        )
        assert corr_check.status == CheckStatus.PASS


class TestValidationResult:
    def test_all_pass(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ]
        )
        df = pl.DataFrame({"x": [0.1, -0.2, 0.3]})
        result = validate(df, profile)
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.summary["fail"] == 0

    def test_per_column_overrides(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 50, "std": 10, "min": 20.0, "max": 80.0},
                    validation_overrides={"range_tolerance": 10.0},
                )
            ]
        )
        # Even with wildly out-of-range data, a very loose tolerance passes
        df = pl.DataFrame({"x": [0.0, 50.0, 200.0]})
        result = validate(df, profile)
        range_check = next(
            (c for c in result.checks if c.check_name == "value_range"), None
        )
        assert range_check is not None
        assert range_check.status == CheckStatus.PASS
