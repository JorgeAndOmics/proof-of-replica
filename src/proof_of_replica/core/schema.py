"""Top-level profile schema and configuration models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.enums import CorrelationMethod, MissingnessPattern

# ── Global config blocks ─────────────────────────────────────


class DefaultsConfig(BaseModel):
    """Global generation defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    noise_level: float = Field(default=0.05, ge=0.0, le=1.0)
    preserve_nulls: bool = True


class ProfilerConfig(BaseModel):
    """Thresholds used by the profiling engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    categorical_max_cardinality: int = 50
    categorical_max_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    distribution_fit_pvalue: float = Field(default=0.05, ge=0.0, le=1.0)


class ValidationConfig(BaseModel):
    """Thresholds used by the validation engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ks_pvalue: float = Field(default=0.05, ge=0.0, le=1.0)
    chisq_pvalue: float = Field(default=0.05, ge=0.0, le=1.0)
    null_tolerance: float = Field(default=0.02, ge=0.0)
    range_tolerance: float = Field(default=0.05, ge=0.0)
    boolean_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    correlation_frobenius: float = Field(default=0.1, ge=0.0)


# ── Correlation config ───────────────────────────────────────


class CorrelationConfig(BaseModel):
    """Pairwise correlation matrix for numeric columns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: list[str]
    matrix: list[list[float]]
    method: CorrelationMethod = CorrelationMethod.PEARSON

    @model_validator(mode="after")
    def _validate_matrix(self) -> "CorrelationConfig":
        """Check that the matrix is square and matches column count."""
        n = len(self.columns)
        if len(self.matrix) != n:
            msg = f"Matrix row count ({len(self.matrix)}) must match column count ({n})"
            raise ValueError(msg)
        for i, row in enumerate(self.matrix):
            if len(row) != n:
                msg = f"Matrix row {i} has {len(row)} elements; expected {n} (must be square)"
                raise ValueError(msg)
        return self


# ── Missingness config ───────────────────────────────────────


class MissingnessConfig(BaseModel):
    """Missingness pattern specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: MissingnessPattern
    co_missing: dict[str, float] | None = None


# ── Top-level Profile ────────────────────────────────────────


class Profile(BaseModel):
    """The top-level profile schema for proof-of-replica.

    This is the single source of truth for replica generation.
    All configuration lives here; CLI flags set initial values
    during profiling, but the persisted profile is canonical.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str = "proof-of-replica"
    version: str
    created_at: str | None = None
    source_hash: str | None = None
    seed: int = 42
    row_count: int | None = None

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    columns: list[ColumnDefinition] = Field(min_length=1)

    correlations: CorrelationConfig | None = None
    missingness: MissingnessConfig | None = None
