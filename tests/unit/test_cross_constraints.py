"""Tests for cross-column constraint schema models."""

import pytest
from pydantic import ValidationError

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.cross_constraints import (
    ArithmeticInvariant,
    CompositeUniqueness,
    ConditionalNull,
    ConstraintAction,
    CrossConstraintBlock,
    TemporalOrdering,
)
from proof_of_replica.core.schema import Profile


class TestConditionalNull:
    def test_construction(self):
        c = ConditionalNull(column="dosage", condition="treatment == 'placebo'")
        assert c.action == ConstraintAction.SET_NULL

    def test_with_value(self):
        c = ConditionalNull(
            column="x", condition="y > 0", action="set_value", value=0.0
        )
        assert c.value == 0.0

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            ConditionalNull(column="x", condition="y > 0", bogus=True)  # type: ignore[call-arg]


class TestTemporalOrdering:
    def test_construction(self):
        t = TemporalOrdering(before="admission_date", after="discharge_date")
        assert t.before == "admission_date"


class TestArithmeticInvariant:
    def test_construction(self):
        a = ArithmeticInvariant(derived="bmi", expression="weight / (height/100)**2")
        assert a.tolerance == 0.01

    def test_custom_tolerance(self):
        a = ArithmeticInvariant(derived="total", expression="a + b", tolerance=0.1)
        assert a.tolerance == 0.1


class TestCompositeUniqueness:
    def test_construction(self):
        c = CompositeUniqueness(columns=["patient_id", "visit_number"])
        assert c.unique is True

    def test_rejects_single_column(self):
        with pytest.raises(ValidationError):
            CompositeUniqueness(columns=["x"])


class TestCrossConstraintBlock:
    def test_empty_block(self):
        block = CrossConstraintBlock()
        assert block.conditional_nulls == []
        assert block.temporal_ordering == []

    def test_full_block(self):
        block = CrossConstraintBlock(
            conditional_nulls=[
                ConditionalNull(column="x", condition="y == 0"),
            ],
            temporal_ordering=[
                TemporalOrdering(before="start", after="end"),
            ],
            arithmetic_invariants=[
                ArithmeticInvariant(derived="c", expression="a + b"),
            ],
            composite_uniqueness=[
                CompositeUniqueness(columns=["a", "b"]),
            ],
        )
        assert len(block.conditional_nulls) == 1
        assert len(block.composite_uniqueness) == 1

    def test_json_round_trip(self):
        block = CrossConstraintBlock(
            conditional_nulls=[
                ConditionalNull(column="x", condition="y == 0"),
            ],
        )
        data = block.model_dump()
        restored = CrossConstraintBlock.model_validate(data)
        assert restored == block


class TestProfileWithConstraints:
    def test_profile_with_constraints(self):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
            constraints=CrossConstraintBlock(
                conditional_nulls=[
                    ConditionalNull(column="x", condition="x > 0"),
                ],
            ),
        )
        assert p.constraints is not None
        assert len(p.constraints.conditional_nulls) == 1

    def test_profile_without_constraints(self):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        assert p.constraints is None

    def test_profile_with_hooks(self):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
            hooks={"post_generate": "hooks/my_hook.py"},
        )
        assert p.hooks is not None
        assert p.hooks["post_generate"] == "hooks/my_hook.py"
