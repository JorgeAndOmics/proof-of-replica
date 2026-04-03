"""Top-level profile schema and configuration models."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.cross_constraints import CrossConstraintBlock
from proof_of_replica.core.enums import CorrelationMethod, MissingnessPattern
from proof_of_replica.exceptions import (
    ProfileError,
    ProfileIssue,
    ProfileValidationError,
)

logger = logging.getLogger(__name__)

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
    constraints: CrossConstraintBlock | None = None
    hooks: dict[str, str] | None = None


# ── Profile I/O ──────────────────────────────────────────────


def _pydantic_error_to_issues(exc: ValidationError) -> list[ProfileIssue]:
    """Convert Pydantic ValidationError to structured ProfileIssues."""
    issues: list[ProfileIssue] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        problem = error["msg"]
        suggestion = f"Check the value at '{location}' in your profile JSON."
        issues.append(
            ProfileIssue(
                location=location,
                problem=problem,
                suggestion=suggestion,
            )
        )
    return issues


def load_profile(path: Path) -> Profile:
    """Load and validate a profile from a JSON file.

    Args:
        path: Path to the profile JSON file.

    Returns:
        A validated Profile instance.

    Raises:
        ProfileError: If the file cannot be read or parsed as JSON.
        ProfileValidationError: If the JSON fails schema validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot read profile file: {path}"
        raise ProfileError(msg) from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in profile file: {path}"
        raise ProfileError(msg) from exc

    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        issues = _pydantic_error_to_issues(exc)
        msg = f"Profile validation failed with {len(issues)} error(s) in {path}"
        raise ProfileValidationError(msg, issues=issues) from exc


def save_profile(profile: Profile, path: Path) -> None:
    """Serialize a profile to a JSON file.

    Args:
        profile: The Profile instance to save.
        path: Output file path.
    """
    data = profile.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def generate_json_schema() -> dict[str, Any]:
    """Generate the JSON Schema for the Profile model.

    Returns:
        A dict representing the JSON Schema, suitable for writing
        to ``profile.schema.json``.
    """
    return Profile.model_json_schema()


def merge_overrides(profile: Profile, overrides: dict[str, Any]) -> Profile:
    """Merge override values onto a profile, returning a new Profile.

    Performs a shallow merge at the top level. For the ``columns`` key,
    merges per-column by matching on column name.

    Args:
        profile: The base profile.
        overrides: Dict of values to merge (same structure as profile JSON).

    Returns:
        A new validated Profile with overrides applied.

    Raises:
        ProfileValidationError: If the merged result fails validation.
    """
    base = profile.model_dump(mode="json", exclude_none=True)

    column_overrides = overrides.pop("columns", None)
    base.update(overrides)

    if column_overrides is not None and isinstance(column_overrides, list):
        col_map = {c["name"]: i for i, c in enumerate(base["columns"])}
        for col_override in column_overrides:
            name = col_override.get("name")
            if name and name in col_map:
                idx = col_map[name]
                base["columns"][idx].update(col_override)

    try:
        return Profile.model_validate(base)
    except ValidationError as exc:
        issues = _pydantic_error_to_issues(exc)
        msg = f"Override merge failed with {len(issues)} error(s)"
        raise ProfileValidationError(msg, issues=issues) from exc
