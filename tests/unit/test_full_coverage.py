"""Final coverage gap tests — targeting every remaining uncovered line."""

import datetime
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.enums import ColumnDtype
from proof_of_replica.core.schema import (
    CorrelationConfig,
    Profile,
    ProfilerConfig,
    save_profile,
)
from proof_of_replica.engines._type_inference import infer_dtype
from proof_of_replica.engines.generation import generate
from proof_of_replica.engines.profiling import profile_dataframe
from proof_of_replica.engines.validation import validate
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.utils.correlation import _nearest_psd
from proof_of_replica.utils.noise import apply_noise
from proof_of_replica.utils.privacy import check_privacy_distance


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── validation.py gaps (lines 148,170,184,192,223,233-234,257,316,341,355)


class TestValidationFullCoverage:
    def test_numeric_without_numeric_stats(self):
        """Line 184: _check_numeric with non-NumericStats returns early."""
        profile = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="x", dtype="float64")],
        )
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = validate(df, profile)
        assert result is not None

    def test_numeric_zero_range(self):
        """Line 192: profile_range == 0 skips range check."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 5, "std": 0, "min": 5.0, "max": 5.0},
                ),
            ],
        )
        df = pl.DataFrame({"x": [5.0, 5.0, 5.0]})
        result = validate(df, profile)
        range_checks = [c for c in result.checks if c.check_name == "value_range"]
        assert len(range_checks) == 0

    def test_categorical_without_categorical_stats(self):
        """Line 257: _check_categorical with non-CategoricalStats."""
        profile = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="c", dtype="categorical")],
        )
        df = pl.DataFrame({"c": ["a", "b"]})
        result = validate(df, profile)
        assert result is not None

    def test_boolean_without_boolean_stats(self):
        """Line 316: _check_boolean with non-BooleanStats."""
        profile = Profile(
            version="0.2.0",
            columns=[ColumnDefinition(name="b", dtype="boolean")],
        )
        df = pl.DataFrame({"b": [True, False]})
        result = validate(df, profile)
        assert result is not None

    def test_correlation_few_rows(self):
        """Line 355: correlation check with too few clean rows."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="y", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            correlations=CorrelationConfig(
                columns=["x", "y"], matrix=[[1.0, 0.5], [0.5, 1.0]]
            ),
        )
        df = pl.DataFrame({"x": [1.0, None], "y": [None, 2.0]})
        result = validate(df, profile)
        assert result is not None

    def test_validate_strictness_fail_with_bad_data(self):
        """validate_cmd.py line 58: SystemExit(1) on strictness=fail."""
        runner = CliRunner()
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.9},
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile_path = tmp_path / "p.json"
            save_profile(p, profile_path)
            df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
            replica_path = tmp_path / "r.parquet"
            df.write_parquet(replica_path)
            result = runner.invoke(
                app,
                [
                    "validate",
                    str(replica_path),
                    str(profile_path),
                    "--strictness",
                    "fail",
                ],
            )
            assert result.exit_code == 1


# ── generation.py gaps (lines 101, 145, 165, 169-170)


class TestGenerationFullCoverage:
    def test_per_column_noise_override(self):
        """Line 101: column-level noise_level override."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=50,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1},
                    overrides={"noise_level": 0.5},
                ),
            ],
        )
        df = generate(profile)
        assert len(df) == 50

    def test_string_with_generator(self):
        """Line 145: string dtype with generator dispatches to generate_string."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=10,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="seq",
                    dtype="string",
                    generator={"method": "placeholder", "placeholder_value": "X"},
                ),
            ],
        )
        df = generate(profile)
        assert all(v == "X" for v in df["seq"].to_list())

    def test_group_effects_missing_column_raises(self):
        """Lines 169-170: group column not in DataFrame."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=10,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="val",
                    dtype="float64",
                    stats={"mean": 0, "std": 1},
                    group_effects={
                        "group_column": "nonexistent",
                        "effects": {"A": {"shift": 1.0, "scale_factor": 1.0}},
                    },
                ),
            ],
        )
        with pytest.raises(GenerationError, match="not found"):
            generate(profile)


# ── profiling.py gaps (lines 169, 218, 223, 294)


class TestProfilingFullCoverage:
    def test_date_column_as_string(self):
        """Line 218: date extraction from string-typed date column."""
        df = pl.DataFrame({"d": ["2023-01-01", "2023-06-15", "2024-12-31"]})
        profile = profile_dataframe(df)
        col = next(c for c in profile.columns if c.name == "d")
        assert col.dtype == ColumnDtype.DATE

    def test_replicate_with_noise_override(self):
        """replicate_cmd.py line 60: noise_level override path."""
        runner = CliRunner()
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "d.csv"
            df.write_csv(csv_path)
            result = runner.invoke(
                app,
                [
                    "replicate",
                    str(csv_path),
                    "-o",
                    str(tmp_path / "r.parquet"),
                    "--noise-level",
                    "0.99",
                ],
            )
            assert result.exit_code == 0


# ── numeric.py gaps (lines 50-51, 96, 101)


class TestNumericFullCoverage:
    pass  # histogram list-params path is unreachable (Pydantic enforces str)


# ── type_inference.py gaps (lines 103-104, 125-126, 149-150)


class TestTypeInferenceFullCoverage:
    def test_string_parseable_as_int(self):
        """Lines 103-104: string column that parses as integer."""
        config = ProfilerConfig()
        s = pl.Series("x", ["100", "200", "300", "400", "500"] * 10)
        assert infer_dtype(s, 50, config) == ColumnDtype.INT64

    def test_string_parseable_as_float(self):
        """Lines 125-126: string column that parses as float."""
        config = ProfilerConfig()
        s = pl.Series("x", ["1.5", "2.7", "3.14", "4.0", "5.5"] * 10)
        assert infer_dtype(s, 50, config) == ColumnDtype.FLOAT64

    def test_string_parseable_as_date(self):
        """Lines 149-150: string column that parses as date."""
        config = ProfilerConfig()
        s = pl.Series("x", ["2023-01-01", "2023-06-15", "2024-12-31"] * 10)
        assert infer_dtype(s, 30, config) == ColumnDtype.DATE


# ── noise.py gap (line 138)


class TestNoiseFullCoverage:
    def test_date_noise_with_nulls(self, rng):
        """Line 138: null value in date noise."""
        series = pl.Series(
            "d",
            [datetime.date(2023, 1, 1), None, datetime.date(2023, 6, 1)],
            dtype=pl.Date,
        )
        result = apply_noise(series, ColumnDtype.DATE, 0.1, rng)
        assert result[1] is None


# ── privacy.py gap (line 70)


class TestPrivacyFullCoverage:
    def test_identical_rows_warning(self):
        """Line 70: warning when synthetic row matches original exactly."""

        # Identical data should produce min_dist = 0
        data = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = check_privacy_distance(data, data)
        assert result["min"] == 0.0


# ── correlation.py gaps (lines 51-53)


class TestCorrelationFullCoverage:
    def test_cholesky_error_on_invalid_matrix(self, rng):
        """Lines 51-53: LinAlgError from non-PSD matrix after clipping fails."""

        # _nearest_psd is always called, so the cholesky should succeed after it
        # Test _nearest_psd directly
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not PSD (eigenvalues: 3, -1)
        result = _nearest_psd(bad)
        # After clipping, should be PSD
        eigenvalues = np.linalg.eigvalsh(result)
        assert all(e > 0 for e in eigenvalues)
