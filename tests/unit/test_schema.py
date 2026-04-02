"""Tests for top-level profile schema models."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from proof_of_replica.core.column_schema import ColumnDefinition, NumericStats
from proof_of_replica.core.enums import CorrelationMethod, MissingnessPattern
from proof_of_replica.core.schema import (
    CorrelationConfig,
    DefaultsConfig,
    MissingnessConfig,
    Profile,
    ProfilerConfig,
    ValidationConfig,
    generate_json_schema,
    load_profile,
    save_profile,
)
from proof_of_replica.exceptions import (
    ProfileError,
    ProfileIssue,
    ProfileValidationError,
)

# ── DefaultsConfig ───────────────────────────────────────────


class TestDefaultsConfig:
    def test_defaults(self):
        d = DefaultsConfig()
        assert d.noise_level == 0.05
        assert d.preserve_nulls is True

    def test_noise_level_range(self):
        with pytest.raises(ValidationError):
            DefaultsConfig(noise_level=1.5)


# ── ProfilerConfig ───────────────────────────────────────────


class TestProfilerConfig:
    def test_defaults(self):
        p = ProfilerConfig()
        assert p.categorical_max_cardinality == 50
        assert p.categorical_max_fraction == 0.05
        assert p.correlation_threshold == 0.05
        assert p.distribution_fit_pvalue == 0.05

    def test_custom_values(self):
        p = ProfilerConfig(categorical_max_cardinality=100)
        assert p.categorical_max_cardinality == 100


# ── ValidationConfig ─────────────────────────────────────────


class TestValidationConfig:
    def test_defaults(self):
        v = ValidationConfig()
        assert v.ks_pvalue == 0.05
        assert v.chisq_pvalue == 0.05
        assert v.null_tolerance == 0.02
        assert v.range_tolerance == 0.05
        assert v.boolean_tolerance == 0.05
        assert v.correlation_frobenius == 0.1

    def test_value_range(self):
        with pytest.raises(ValidationError):
            ValidationConfig(ks_pvalue=2.0)


# ── CorrelationConfig ────────────────────────────────────────


class TestCorrelationConfig:
    def test_valid_construction(self):
        c = CorrelationConfig(
            columns=["age", "gene_expr"],
            matrix=[[1.0, -0.12], [-0.12, 1.0]],
        )
        assert c.method == CorrelationMethod.PEARSON
        assert len(c.matrix) == 2

    def test_matrix_size_mismatch(self):
        with pytest.raises(ValidationError, match="must match"):
            CorrelationConfig(
                columns=["a", "b", "c"],
                matrix=[[1.0, 0.5], [0.5, 1.0]],
            )

    def test_non_square_matrix(self):
        with pytest.raises(ValidationError, match="square"):
            CorrelationConfig(
                columns=["a", "b"],
                matrix=[[1.0, 0.5, 0.3], [0.5, 1.0, 0.2]],
            )


# ── MissingnessConfig ────────────────────────────────────────


class TestMissingnessConfig:
    def test_construction(self):
        m = MissingnessConfig(
            pattern=MissingnessPattern.MCAR,
            co_missing={"notes,measurement_date": 0.8},
        )
        assert m.pattern == "MCAR"

    def test_minimal(self):
        m = MissingnessConfig(pattern="observed")
        assert m.co_missing is None


# ── Profile ──────────────────────────────────────────────────


class TestProfile:
    def test_minimal_construction(self):
        p = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="id", dtype="string", role="identifier")],
        )
        assert p.tool == "proof-of-replica"
        assert p.seed == 42
        assert p.defaults is not None
        assert p.defaults.noise_level == 0.05

    def test_rejects_empty_columns(self):
        with pytest.raises(ValidationError, match="at least 1"):
            Profile(version="0.2.0", columns=[])

    def test_defaults_populated(self):
        p = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="x", dtype="float64")],
        )
        assert p.defaults == DefaultsConfig()
        assert p.profiler == ProfilerConfig()
        assert p.validation == ValidationConfig()

    def test_full_construction(self):
        p = Profile(
            version="0.2.0",
            seed=123,
            row_count=1000,
            columns=[
                ColumnDefinition(
                    name="age",
                    dtype="float64",
                    stats={"mean": 52.3, "std": 14.7},
                    distribution={
                        "family": "normal",
                        "params": {"loc": 52.3, "scale": 14.7},
                    },
                ),
                ColumnDefinition(
                    name="condition",
                    dtype="categorical",
                    role="group",
                    stats={"cardinality": 3, "value_counts": {"a": 0.5, "b": 0.5}},
                ),
            ],
            correlations=CorrelationConfig(
                columns=["age"],
                matrix=[[1.0]],
            ),
            missingness=MissingnessConfig(pattern="MCAR"),
        )
        assert p.row_count == 1000
        assert len(p.columns) == 2
        assert p.correlations is not None
        assert p.missingness is not None

    def test_json_round_trip(self):
        p = Profile(
            version="0.2.0",
            seed=42,
            row_count=1500,
            columns=[
                ColumnDefinition(
                    name="sample_id",
                    dtype="string",
                    role="identifier",
                    generator={"method": "sequential", "prefix": "S_", "zero_pad": 4},
                ),
                ColumnDefinition(
                    name="age",
                    dtype="float64",
                    stats={"mean": 52.3, "std": 14.7, "min": 18.0, "max": 89.0},
                ),
            ],
        )
        data = p.model_dump()
        restored = Profile.model_validate(data)
        assert restored == p
        assert isinstance(restored.columns[1].stats, NumericStats)

    def test_from_json_string(self):
        json_str = (
            '{"version": "0.2.0", "columns": ['
            '{"name": "x", "dtype": "float64", "stats": {"mean": 1.0, "std": 0.5}}'
            "]}"
        )
        p = Profile.model_validate_json(json_str)
        assert p.tool == "proof-of-replica"
        assert len(p.columns) == 1


# ── load_profile / save_profile ──────────────────────────────


class TestLoadProfile:
    def test_load_valid(self, tmp_path: Path):
        data = {
            "version": "0.2.0",
            "columns": [{"name": "x", "dtype": "float64"}],
        }
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(data))
        p = load_profile(path)
        assert p.version == "0.2.0"

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(ProfileError, match="Cannot read"):
            load_profile(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(ProfileError, match="Invalid JSON"):
            load_profile(path)

    def test_load_schema_invalid(self, tmp_path: Path):
        data = {"version": "0.2.0", "columns": []}
        path = tmp_path / "empty_cols.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ProfileValidationError) as exc_info:
            load_profile(path)
        assert len(exc_info.value.issues) > 0
        assert exc_info.value.issues[0].location


class TestSaveProfile:
    def test_round_trip(self, tmp_path: Path):
        p = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="id", dtype="string", role="identifier")],
        )
        path = tmp_path / "out.json"
        save_profile(p, path)

        loaded = load_profile(path)
        assert loaded.version == p.version
        assert loaded.columns[0].name == "id"

    def test_excludes_none(self, tmp_path: Path):
        p = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="x", dtype="float64")],
        )
        path = tmp_path / "out.json"
        save_profile(p, path)
        data = json.loads(path.read_text())
        assert "source_hash" not in data
        assert "correlations" not in data


# ── generate_json_schema ─────────────────────────────────────


class TestGenerateJsonSchema:
    def test_returns_dict(self):
        schema = generate_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "columns" in schema["properties"]

    def test_schema_has_title(self):
        schema = generate_json_schema()
        assert schema.get("title") == "Profile"


# ── ProfileValidationError ───────────────────────────────────


class TestProfileValidationError:
    def test_carries_issues(self):
        issues = [
            ProfileIssue(
                location="columns[0].stats.mean",
                problem="missing required field",
                suggestion="Add 'mean' to the stats block.",
            ),
        ]
        exc = ProfileValidationError("test", issues=issues)
        assert len(exc.issues) == 1
        assert exc.issues[0].severity == "error"

    def test_empty_issues_by_default(self):
        exc = ProfileValidationError("test")
        assert exc.issues == []
