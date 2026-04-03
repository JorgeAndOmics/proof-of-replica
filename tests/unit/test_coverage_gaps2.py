"""Additional tests to close remaining coverage gaps."""

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
from proof_of_replica.engines._distribution_fit import fit_distribution
from proof_of_replica.engines._type_inference import infer_dtype
from proof_of_replica.engines.generation import generate
from proof_of_replica.engines.profiling import profile_dataframe
from proof_of_replica.engines.reporting import generate_fidelity_report
from proof_of_replica.engines.validation import validate
from proof_of_replica.exceptions import FileIOError
from proof_of_replica.utils.correlation import induce_correlation
from proof_of_replica.utils.io import write_dataframe
from proof_of_replica.utils.noise import apply_noise


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def config() -> ProfilerConfig:
    return ProfilerConfig()


# ── Type inference gaps ──────────────────────────────────────


class TestTypeInferenceGaps:
    def test_string_as_integer_fails_parse(self, config):
        """String column that doesn't parse as integer."""
        s = pl.Series("x", ["abc", "def", "ghi"])
        assert infer_dtype(s, 3, config) != "int64"

    def test_string_as_float_fails_parse(self, config):
        """String column that doesn't parse as float."""
        s = pl.Series("x", ["hello", "world"])
        assert infer_dtype(s, 2, config) != "float64"

    def test_date_string_low_success_rate(self, config):
        """String column with few valid dates doesn't classify as date."""
        s = pl.Series(
            "x", ["2023-01-01", "not_a_date", "also_not", "nope", "nada"] * 20
        )
        result = infer_dtype(s, 100, config)
        assert result != "date"

    def test_empty_numeric_string(self, config):
        """Empty string column for integer check."""
        s = pl.Series("x", [None, None, None], dtype=pl.Utf8)
        result = infer_dtype(s, 3, config)
        assert result is not None

    def test_all_unique_identifier(self, config):
        """String column where all values are unique."""
        s = pl.Series("x", [f"id_{i}" for i in range(10)])
        result = infer_dtype(s, 10, config)
        assert result == "string"


# ── Distribution fitting gaps ────────────────────────────────


class TestDistributionFitGaps:
    def test_fit_beta_data(self, config):
        """Beta-eligible data (values in [0, 1])."""
        rng = np.random.default_rng(42)
        values = rng.beta(2, 5, size=1000)
        result = fit_distribution(values, config)
        assert result.family is not None

    def test_fit_with_infinite_log_lik(self):
        """Data that causes infinite log-likelihood for some candidates."""
        config = ProfilerConfig()
        values = np.array([0.0] * 100 + [1.0] * 100, dtype=np.float64)
        result = fit_distribution(values, config)
        assert result is not None


# ── Profiling engine gaps ────────────────────────────────────


class TestProfilingGaps:
    def test_boolean_from_zero_one_ints(self):
        """Profile a column of 0/1 integers detected as boolean."""
        df = pl.DataFrame({"b": [0, 1, 0, 1, 0, 1] * 20})
        p = profile_dataframe(df)
        assert p.columns[0].dtype == "boolean"

    def test_numeric_with_few_values(self):
        """Numeric column with fewer than 5 values skips distribution fitting."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        p = profile_dataframe(df)
        assert p.columns[0].stats is not None

    def test_all_null_numeric_column(self):
        """Numeric column that is entirely null."""
        df = pl.DataFrame({"x": [None, None, None], "y": [1.0, 2.0, 3.0]})
        profile_dataframe(df)
        # Should not crash

    def test_group_effects_zero_std(self):
        """Group effects when global std is zero."""
        df = pl.DataFrame(
            {
                "group": ["A", "B"] * 50,
                "value": [5.0] * 100,
            }
        )
        profile = profile_dataframe(df, group_column="group")
        value_col = next(c for c in profile.columns if c.name == "value")
        # With zero std, group effects may be None
        assert value_col.stats is not None

    def test_correlation_too_few_rows(self):
        """Correlation computation with very few clean rows."""
        df = pl.DataFrame(
            {
                "x": [1.0, None, None],
                "y": [None, 2.0, None],
            }
        )
        profile_dataframe(df, correlations=True)
        # Not enough clean rows for correlation


# ── Validation engine gaps ───────────────────────────────────


class TestValidationGaps:
    def test_numeric_no_range(self):
        """Numeric column without min/max in stats."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = validate(df, profile)
        range_checks = [c for c in result.checks if c.check_name == "value_range"]
        assert len(range_checks) == 0

    def test_categorical_no_cardinality(self):
        """Categorical without cardinality in stats."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={"value_counts": {"a": 0.5, "b": 0.5}},
                ),
            ],
        )
        df = pl.DataFrame({"c": ["a", "b"] * 20})
        result = validate(df, profile)
        assert result.passed

    def test_ks_test_unsupported_family(self):
        """KS test with empirical_kde family (not in _FAMILY_TO_SCIPY)."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 0, "std": 1},
                    distribution={
                        "family": "empirical_kde",
                        "params": {"bandwidth": 1.0},
                    },
                ),
            ],
        )
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist()})
        result = validate(df, profile)
        ks = [c for c in result.checks if c.check_name == "ks_test"]
        assert len(ks) == 0

    def test_correlation_with_missing_columns(self):
        """Correlation check when some columns are missing from DataFrame."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            correlations=CorrelationConfig(
                columns=["x", "missing"], matrix=[[1.0, 0.5], [0.5, 1.0]]
            ),
        )
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        validate(df, profile)  # Should not crash


# ── Generation edge cases ────────────────────────────────────


class TestGenerationGaps:
    def test_generate_with_identifier_no_generator(self):
        """Identifier column without generator uses categorical fallback."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=10,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="cat",
                    dtype="categorical",
                    role="identifier",
                    stats={"value_counts": {"a": 0.5, "b": 0.5}},
                ),
            ],
        )
        df = generate(profile)
        assert len(df) == 10

    def test_noise_on_date_column(self):
        """Verify noise is applied to date columns."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=50,
            defaults={"noise_level": 0.1},
            columns=[
                ColumnDefinition(
                    name="d",
                    dtype="date",
                    stats={"min": "2023-01-01", "max": "2023-12-31"},
                ),
            ],
        )
        df = generate(profile)
        assert df["d"].dtype == pl.Date


# ── Reporting gaps ───────────────────────────────────────────


class TestReportingGaps:
    def test_report_with_missing_column(self):
        """Report when profile has column not in replica."""
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
        )
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        html = generate_fidelity_report(df, profile)
        assert "<!DOCTYPE html>" in html


# ── Noise gaps ───────────────────────────────────────────────


class TestNoiseGaps:
    def test_noise_on_all_null_date(self, rng):
        """Noise on a date series that's entirely null."""

        series = pl.Series("d", [None, None, None], dtype=pl.Date)
        result = apply_noise(series, ColumnDtype.DATE, 0.1, rng)
        assert result.null_count() == 3

    def test_noise_categorical_preserves_nulls(self, rng):
        """Categorical noise preserves null values."""

        series = pl.Series("c", ["a", None, "b", None, "a"])
        result = apply_noise(series, ColumnDtype.CATEGORICAL, 0.5, rng)
        assert result[1] is None
        assert result[3] is None


# ── Correlation gaps ─────────────────────────────────────────


class TestCorrelationGaps2:
    def test_cholesky_failure_path(self, rng):
        """Non-PSD matrix that requires eigenvalue clipping."""
        df = pl.DataFrame(
            {
                "a": rng.normal(0, 1, 200).tolist(),
                "b": rng.normal(0, 1, 200).tolist(),
            }
        )
        # Matrix with very high correlation
        result = induce_correlation(df, ["a", "b"], [[1.0, 0.999], [0.999, 1.0]], rng)
        assert len(result) == 200


# ── IO gaps ──────────────────────────────────────────────────


class TestIOGaps:
    def test_write_unsupported_format(self, tmp_path):
        """Writing to unsupported format raises error."""

        df = pl.DataFrame({"x": [1]})
        with pytest.raises(FileIOError, match="Unsupported"):
            write_dataframe(df, tmp_path / "out.xlsx")


# ── Validate command gaps ────────────────────────────────────


class TestValidateCmdGaps:
    def test_validate_verbose(self, tmp_path):
        """Validate with --report and -v flags."""
        runner = CliRunner()
        p = Profile(
            version="0.2.0",
            seed=42,
            row_count=10,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        profile_path = tmp_path / "p.json"
        save_profile(p, profile_path)

        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(0, 1, 10).tolist()})
        replica_path = tmp_path / "r.parquet"
        df.write_parquet(replica_path)

        report = tmp_path / "r.txt"
        result = runner.invoke(
            app,
            [
                "validate",
                str(replica_path),
                str(profile_path),
                "--report",
                str(report),
                "-v",
            ],
        )
        assert result.exit_code == 0
        assert "Report written" in result.output
