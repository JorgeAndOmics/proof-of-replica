"""Tests for cross-column constraint enforcement engine."""

import datetime

import polars as pl
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.cross_constraints import (
    ArithmeticInvariant,
    CompositeUniqueness,
    ConditionalNull,
    CrossConstraintBlock,
    TemporalOrdering,
)
from proof_of_replica.core.schema import Profile
from proof_of_replica.engines.cross_constraints import apply_cross_constraints
from proof_of_replica.engines.generation import generate
from proof_of_replica.exceptions import GenerationError


class TestConditionalNull:
    def test_set_null(self):
        df = pl.DataFrame(
            {
                "treatment": ["drug", "placebo", "drug", "placebo"],
                "dosage": [10.0, 20.0, 30.0, 40.0],
            }
        )
        constraints = CrossConstraintBlock(
            conditional_nulls=[
                ConditionalNull(
                    column="dosage",
                    condition="treatment == 'placebo'",
                    action="set_null",
                ),
            ]
        )
        result = apply_cross_constraints(df, constraints)
        assert result["dosage"][1] is None
        assert result["dosage"][3] is None
        assert result["dosage"][0] == 10.0

    def test_set_zero(self):
        df = pl.DataFrame({"flag": [True, False], "value": [1.0, 2.0]})
        constraints = CrossConstraintBlock(
            conditional_nulls=[
                ConditionalNull(
                    column="value", condition="flag == false", action="set_zero"
                ),
            ]
        )
        result = apply_cross_constraints(df, constraints)
        assert result["value"][1] == 0.0

    def test_missing_column_skipped(self):
        df = pl.DataFrame({"x": [1.0]})
        constraints = CrossConstraintBlock(
            conditional_nulls=[
                ConditionalNull(column="missing", condition="x > 0"),
            ]
        )
        result = apply_cross_constraints(df, constraints)
        assert len(result) == 1

    def test_invalid_condition_raises(self):
        df = pl.DataFrame({"x": [1.0]})
        constraints = CrossConstraintBlock(
            conditional_nulls=[
                ConditionalNull(column="x", condition="INVALID SQL GARBAGE !!!"),
            ]
        )
        with pytest.raises(GenerationError, match="Failed to evaluate"):
            apply_cross_constraints(df, constraints)


class TestTemporalOrdering:
    def test_swaps_violated_rows(self):
        df = pl.DataFrame(
            {
                "start": [datetime.date(2023, 6, 1), datetime.date(2023, 12, 1)],
                "end": [datetime.date(2023, 12, 1), datetime.date(2023, 1, 1)],
            }
        )
        constraints = CrossConstraintBlock(
            temporal_ordering=[
                TemporalOrdering(before="start", after="end"),
            ]
        )
        result = apply_cross_constraints(df, constraints)
        # Row 1 should be swapped
        assert result["start"][1] <= result["end"][1]

    def test_missing_columns_skipped(self):
        df = pl.DataFrame({"x": [1]})
        constraints = CrossConstraintBlock(
            temporal_ordering=[TemporalOrdering(before="a", after="b")]
        )
        result = apply_cross_constraints(df, constraints)
        assert len(result) == 1


class TestArithmeticInvariant:
    def test_recomputes_derived(self):
        df = pl.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": [0.0, 0.0, 0.0]}
        )
        constraints = CrossConstraintBlock(
            arithmetic_invariants=[
                ArithmeticInvariant(derived="c", expression="a + b"),
            ]
        )
        result = apply_cross_constraints(df, constraints)
        assert result["c"].to_list() == [11.0, 22.0, 33.0]

    def test_invalid_expression_raises(self):
        df = pl.DataFrame({"a": [1.0]})
        constraints = CrossConstraintBlock(
            arithmetic_invariants=[
                ArithmeticInvariant(derived="c", expression="NONSENSE!!!"),
            ]
        )
        with pytest.raises(GenerationError, match="Failed to evaluate"):
            apply_cross_constraints(df, constraints)


class TestCompositeUniqueness:
    def test_deduplicates(self):
        df = pl.DataFrame(
            {
                "patient": ["A", "A", "B", "B"],
                "visit": [1, 1, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        constraints = CrossConstraintBlock(
            composite_uniqueness=[
                CompositeUniqueness(columns=["patient", "visit"]),
            ]
        )
        result = apply_cross_constraints(df, constraints)
        assert len(result) == 3  # One duplicate removed

    def test_missing_columns_skipped(self):
        df = pl.DataFrame({"x": [1, 2]})
        constraints = CrossConstraintBlock(
            composite_uniqueness=[CompositeUniqueness(columns=["a", "b"])]
        )
        result = apply_cross_constraints(df, constraints)
        assert len(result) == 2


class TestIntegrationWithPipeline:
    def test_constraints_applied_in_generation(self):
        """Test that constraints are applied during generate()."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=100,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="a",
                    dtype="float64",
                    stats={"mean": 10, "std": 5},
                ),
                ColumnDefinition(
                    name="b",
                    dtype="float64",
                    stats={"mean": 20, "std": 5},
                ),
                ColumnDefinition(
                    name="c",
                    dtype="float64",
                    stats={"mean": 0, "std": 1},
                ),
            ],
            constraints=CrossConstraintBlock(
                arithmetic_invariants=[
                    ArithmeticInvariant(derived="c", expression="a + b"),
                ],
            ),
        )
        df = generate(profile)
        # c should equal a + b
        expected = (df["a"] + df["b"]).to_list()
        actual = df["c"].to_list()
        for e, a in zip(expected, actual, strict=True):
            assert abs(e - a) < 0.01
