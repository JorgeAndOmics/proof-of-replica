"""Column-level schema models for the profile JSON."""

from pydantic import BaseModel, ConfigDict, Field

from proof_of_replica.core.enums import DistributionFamily

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
