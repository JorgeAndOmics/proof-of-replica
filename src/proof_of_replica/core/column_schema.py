"""Column-level schema models for the profile JSON."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proof_of_replica.core.enums import (
    ColumnDtype,
    ColumnRole,
    DistributionFamily,
    GeneratorMethod,
)

# ── Stats models (one per dtype) ─────────────────────────────


class NumericStats(BaseModel):
    """Statistics for float64 and int64 columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    null_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    percentiles: dict[str, float] | None = None


class CategoricalStats(BaseModel):
    """Statistics for categorical columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cardinality: int | None = None
    null_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    value_counts: dict[str, float] = Field(default_factory=dict)


class BooleanStats(BaseModel):
    """Statistics for boolean columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_fraction: float = Field(ge=0.0, le=1.0)
    null_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class DateStats(BaseModel):
    """Statistics for date columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: str | None = None
    max: str | None = None
    null_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class StringStats(BaseModel):
    """Statistics for free-text and structured string columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    null_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_length: float | None = None
    max_length: int | None = None
    pattern_coverage: float | None = Field(default=None, ge=0.0, le=1.0)


# ── Distribution config ─────────────────────────────────────


class DistributionConfig(BaseModel):
    """Parametric or empirical distribution specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: DistributionFamily
    params: dict[str, float | str] = Field(default_factory=dict)


# ── Constraints ──────────────────────────────────────────────


class ColumnConstraints(BaseModel):
    """Per-column value constraints applied after generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: float | None = None
    max: float | None = None
    integer_valued: bool = False


# ── Group effects ────────────────────────────────────────────


class GroupEffect(BaseModel):
    """Shift and scale for a single group level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shift: float = 0.0
    scale_factor: float = 1.0


class GroupEffects(BaseModel):
    """Group-level distribution modifiers for a column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_column: str
    effects: dict[str, GroupEffect]


# ── Per-column overrides ─────────────────────────────────────


class ColumnOverrides(BaseModel):
    """Per-column generation overrides merged over global defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    noise_level: float | None = Field(default=None, ge=0.0, le=1.0)


class ValidationOverrides(BaseModel):
    """Per-column validation threshold overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ks_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)
    chisq_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)
    null_tolerance: float | None = Field(default=None, ge=0.0)
    range_tolerance: float | None = Field(default=None, ge=0.0)
    boolean_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    correlation_frobenius: float | None = Field(default=None, ge=0.0)


# ── Generator config ────────────────────────────────────────


class LengthDistribution(BaseModel):
    """Length specification for alphabet generator strings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distribution: str = "normal"
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


class GeneratorConfig(BaseModel):
    """Column value generator configuration.

    Fields used depend on the generator method. A model validator
    checks that required fields for the chosen method are present.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: GeneratorMethod
    prefix: str | None = None
    zero_pad: int | None = None
    unique: bool = False
    pattern: str | None = None
    chars: str | None = None
    weights: list[float] | None = None
    length: LengthDistribution | None = None
    placeholder_value: str | None = None
    inferred: bool | None = None

    @model_validator(mode="after")
    def _check_method_fields(self) -> "GeneratorConfig":
        """Validate that required fields for the method are present."""
        method = self.method
        if method == GeneratorMethod.REGEX and not self.pattern:
            msg = "Generator method 'regex' requires 'pattern'"
            raise ValueError(msg)
        if method == GeneratorMethod.ALPHABET and not self.chars:
            msg = "Generator method 'alphabet' requires 'chars'"
            raise ValueError(msg)
        return self


# ── Column stats union type ──────────────────────────────────

# Maps dtype -> expected stats model class for validation
_DTYPE_TO_STATS: dict[
    ColumnDtype,
    type[NumericStats | CategoricalStats | BooleanStats | DateStats | StringStats],
] = {
    ColumnDtype.FLOAT64: NumericStats,
    ColumnDtype.INT64: NumericStats,
    ColumnDtype.CATEGORICAL: CategoricalStats,
    ColumnDtype.BOOLEAN: BooleanStats,
    ColumnDtype.DATE: DateStats,
    ColumnDtype.STRING: StringStats,
}

ColumnStats = NumericStats | CategoricalStats | BooleanStats | DateStats | StringStats


# ── Column definition ────────────────────────────────────────


class ColumnDefinition(BaseModel):
    """A single column in the profile schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dtype: ColumnDtype
    role: ColumnRole = ColumnRole.FEATURE
    stats: ColumnStats | None = None
    distribution: DistributionConfig | None = None
    generator: GeneratorConfig | None = None
    constraints: ColumnConstraints | None = None
    group_effects: GroupEffects | None = None
    overrides: ColumnOverrides | None = None
    validation_overrides: ValidationOverrides | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_stats_type(cls, data: Any) -> Any:
        """Parse stats with the correct model based on dtype."""
        if not isinstance(data, dict):  # pragma: no cover
            return data
        raw_stats = data.get("stats")
        raw_dtype = data.get("dtype")
        if raw_stats is None or raw_dtype is None:
            return data
        if isinstance(raw_stats, BaseModel):
            return data
        stats_cls = _DTYPE_TO_STATS[ColumnDtype(raw_dtype)]
        data = dict(data)
        data["stats"] = stats_cls.model_validate(raw_stats)
        return data
