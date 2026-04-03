"""Cross-column constraint schema models for the profile JSON."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ConstraintAction(StrEnum):
    """Action to take when a conditional null constraint fires."""

    SET_NULL = "set_null"
    SET_ZERO = "set_zero"
    SET_VALUE = "set_value"


class ConditionalNull(BaseModel):
    """A column's nullness depends on another column's value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    condition: str
    action: ConstraintAction = ConstraintAction.SET_NULL
    value: str | float | None = None


class TemporalOrdering(BaseModel):
    """One date column must come before another."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    before: str
    after: str


class ArithmeticInvariant(BaseModel):
    """A derived column is computed from an expression over other columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    derived: str
    expression: str
    tolerance: float = 0.01


class CompositeUniqueness(BaseModel):
    """A combination of columns must have unique values per row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: list[str] = Field(min_length=2)
    unique: bool = True


class CrossConstraintBlock(BaseModel):
    """Container for all cross-column constraints in a profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conditional_nulls: list[ConditionalNull] = Field(default_factory=list)
    temporal_ordering: list[TemporalOrdering] = Field(default_factory=list)
    arithmetic_invariants: list[ArithmeticInvariant] = Field(default_factory=list)
    composite_uniqueness: list[CompositeUniqueness] = Field(default_factory=list)
