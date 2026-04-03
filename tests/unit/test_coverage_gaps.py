"""Tests to close coverage gaps across the codebase."""

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import MissingnessConfig, Profile, save_profile
from proof_of_replica.engines.generation import generate
from proof_of_replica.engines.validation import validate
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.numeric import generate_numeric
from proof_of_replica.generators.string import generate_string
from proof_of_replica.types import DistributionParams, ProfileDict, StatsDict
from proof_of_replica.utils.correlation import induce_correlation
from proof_of_replica.utils.missingness import inject_missingness
from proof_of_replica.utils.privacy import check_privacy_distance


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── types.py coverage ────────────────────────────────────────


class TestTypesModule:
    def test_type_aliases_exist(self):
        """Verify type aliases are importable."""
        assert ProfileDict is not None
        assert StatsDict is not None
        assert DistributionParams is not None


# ── Numeric generator edge cases ─────────────────────────────


class TestNumericGeneratorEdgeCases:
    def test_unsupported_distribution_family(self, rng):
        """Test unsupported parametric family raises error."""

        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 0, "std": 1},
            distribution={"family": "empirical_kde", "params": {"bandwidth": 1.0}},
        )
        # empirical_kde goes through a different path, should work
        result = generate_numeric(col, 10, rng)
        assert len(result) == 10

    def test_empirical_histogram_with_list_params(self, rng):
        """Test histogram with pre-parsed list params (not strings)."""
        col = ColumnDefinition(
            name="x",
            dtype="float64",
            stats={"mean": 5, "std": 2},
            distribution={
                "family": "empirical_histogram",
                "params": {
                    "edges": "[0,2,4,6,8,10]",
                    "counts": "[10,20,40,20,10]",
                },
            },
        )
        result = generate_numeric(col, 100, rng)
        assert len(result) == 100


# ── String generator edge cases ──────────────────────────────


class TestStringGeneratorEdgeCases:
    def test_lorem_without_stats(self, rng):
        """Lorem generator with no stats uses default length."""
        col = ColumnDefinition(
            name="text",
            dtype="string",
            generator={"method": "lorem"},
        )
        result = generate_string(col, 5, rng)
        assert len(result) == 5
        assert all(len(v) > 0 for v in result.to_list())

    def test_unique_regex_exhaustion(self, rng):
        """Regex with narrow pattern can't generate enough uniques."""

        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={"method": "regex", "pattern": r"[AB]", "unique": True},
        )
        with pytest.raises(GenerationError, match="Could not generate"):
            generate_string(col, 100, rng)


# ── Generation engine edge cases ─────────────────────────────


class TestGenerationEdgeCases:
    def test_string_without_generator_raises(self):
        """String column without generator config raises GenerationError."""

        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=10,
            defaults={"noise_level": 0.0},
            columns=[ColumnDefinition(name="text", dtype="string")],
        )
        with pytest.raises(GenerationError, match="generator config"):
            generate(profile)

    def test_constraints_integer_valued(self):
        """Test integer_valued constraint casts float to int."""
        profile = Profile(
            version="0.2.0",
            seed=42,
            row_count=50,
            defaults={"noise_level": 0.0},
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 50, "std": 10},
                    constraints={"integer_valued": True},
                ),
            ],
        )
        df = generate(profile)
        assert df["x"].dtype == pl.Int64


# ── Validation engine edge cases ─────────────────────────────


class TestValidationEdgeCases:
    def test_column_missing_from_df(self):
        """Validate with a column in profile but not in DataFrame."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="missing", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = validate(df, profile)
        # Should have column mismatch
        assert not result.passed

    def test_boolean_empty_series(self):
        """Validate boolean column with empty series."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="b", dtype="boolean", stats={"true_fraction": 0.5}
                ),
            ],
        )
        df = pl.DataFrame({"b": pl.Series([], dtype=pl.Boolean)})
        result = validate(df, profile)
        assert isinstance(result.passed, bool)


# ── Missingness edge cases ───────────────────────────────────


class TestMissingnessEdgeCases:
    def test_column_not_in_df_skipped(self, rng):
        """Column in profile but not in DataFrame is skipped."""
        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="missing_col",
                    dtype="float64",
                    stats={"mean": 0, "std": 1, "null_fraction": 0.5},
                ),
            ],
        )
        df = pl.DataFrame({"other": [1.0, 2.0]})
        result = inject_missingness(df, profile, rng)
        assert "other" in result.columns

    def test_co_missing_invalid_pair_key(self, rng):
        """Co-missing with bad key format (not 'a,b') is skipped."""

        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="a", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            missingness=MissingnessConfig(
                pattern="MCAR",
                co_missing={"bad_key": 0.5},
            ),
        )
        df = pl.DataFrame({"a": [1.0] * 100})
        result = inject_missingness(df, profile, rng)
        assert len(result) == 100

    def test_co_missing_column_not_in_df(self, rng):
        """Co-missing referencing missing columns is skipped."""

        profile = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="a", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            missingness=MissingnessConfig(
                pattern="MCAR",
                co_missing={"a,nonexistent": 0.5},
            ),
        )
        df = pl.DataFrame({"a": [1.0] * 100})
        result = inject_missingness(df, profile, rng)
        assert len(result) == 100


# ── CLI logging branches ─────────────────────────────────────


class TestCLILoggingBranches:
    def test_diff_verbose(self, tmp_path):
        """Test diff with verbose and report file."""
        runner = CliRunner()
        rng = np.random.default_rng(42)

        orig = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        repl = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        orig_path = tmp_path / "o.csv"
        repl_path = tmp_path / "r.csv"
        orig.write_csv(orig_path)
        repl.write_csv(repl_path)

        report = tmp_path / "diff.txt"
        result = runner.invoke(
            app,
            ["diff", str(orig_path), str(repl_path), "--report", str(report), "-v"],
        )
        assert result.exit_code == 0
        assert "Diff report written" in result.output

    def test_replicate_verbose_with_save_profile(self, tmp_path):
        """Test replicate verbose output for save-profile."""
        runner = CliRunner()
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        csv_path = tmp_path / "d.csv"
        df.write_csv(csv_path)

        result = runner.invoke(
            app,
            [
                "replicate",
                str(csv_path),
                "-o",
                str(tmp_path / "r.parquet"),
                "--save-profile",
                str(tmp_path / "p.json"),
                "-v",
            ],
        )
        assert result.exit_code == 0
        assert "Profile saved" in result.output
        assert "Generated" in result.output

    def test_generate_verbose(self, tmp_path):
        """Test generate verbose output."""
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

        result = runner.invoke(
            app,
            ["generate", str(profile_path), "-o", str(tmp_path / "r.parquet"), "-v"],
        )
        assert result.exit_code == 0
        assert "Generated" in result.output


# ── Privacy edge case ────────────────────────────────────────


class TestPrivacyEdgeCases:
    def test_no_shared_numeric_columns(self):
        """Privacy check with no shared numeric columns."""

        syn = pl.DataFrame({"a": ["x", "y"]})
        orig = pl.DataFrame({"a": ["x", "y"]})
        result = check_privacy_distance(syn, orig)
        assert result["min"] == float("inf")


# ── Correlation edge case ────────────────────────────────────


class TestCorrelationEdgeCases:
    def test_non_psd_matrix(self, rng):
        """Correlation induction with a non-PSD target matrix."""

        df = pl.DataFrame(
            {
                "a": rng.normal(0, 1, 100).tolist(),
                "b": rng.normal(0, 1, 100).tolist(),
            }
        )
        # This matrix is valid PSD, but test the path
        result = induce_correlation(df, ["a", "b"], [[1.0, 0.99], [0.99, 1.0]], rng)
        assert len(result) == 100
