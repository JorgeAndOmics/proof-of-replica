"""Tests for the diff engine."""

import datetime

import numpy as np
import polars as pl

from proof_of_replica.engines.diff import diff_datasets


class TestDiffNumeric:
    def test_numeric_stats_comparison(self):
        orig = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        repl = pl.DataFrame({"x": [1.5, 2.5, 3.5, 4.5, 5.5]})
        result = diff_datasets(orig, repl)

        assert len(result.column_diffs) == 1
        col = result.column_diffs[0]
        assert col.name == "x"
        assert "mean" in col.stats_comparison
        assert col.stats_comparison["mean"].diff is not None
        assert abs(col.stats_comparison["mean"].diff - 0.5) < 0.01

    def test_numeric_ks_test(self):
        rng = np.random.default_rng(42)
        orig = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist()})
        repl = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist()})
        result = diff_datasets(orig, repl)

        col = result.column_diffs[0]
        assert col.distribution_test is not None
        assert col.distribution_test.test_name == "ks_2samp"


class TestDiffBoolean:
    def test_boolean_comparison(self):
        orig = pl.DataFrame({"b": [True, False] * 50})
        repl = pl.DataFrame({"b": [True, True, False] * 34})
        result = diff_datasets(orig, repl)

        col = result.column_diffs[0]
        assert "true_fraction" in col.stats_comparison


class TestDiffCategorical:
    def test_categorical_comparison(self):
        orig = pl.DataFrame({"c": ["a", "b", "c"] * 30})
        repl = pl.DataFrame({"c": ["a", "b", "c"] * 25})
        result = diff_datasets(orig, repl)

        col = result.column_diffs[0]
        assert "cardinality" in col.stats_comparison

    def test_chisquare_test(self):
        orig = pl.DataFrame({"c": ["a", "b"] * 50})
        repl = pl.DataFrame({"c": ["a", "b"] * 40})
        result = diff_datasets(orig, repl)

        col = result.column_diffs[0]
        assert col.distribution_test is not None
        assert col.distribution_test.test_name == "chisquare"


class TestDiffDate:
    def test_date_comparison(self):
        orig = pl.DataFrame({"d": [datetime.date(2023, 1, i) for i in range(1, 11)]})
        repl = pl.DataFrame({"d": [datetime.date(2023, 1, i) for i in range(5, 15)]})
        result = diff_datasets(orig, repl)

        col = result.column_diffs[0]
        assert "min" in col.stats_comparison
        assert "max" in col.stats_comparison


class TestDiffPrivacy:
    def test_with_privacy(self):
        orig = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        repl = pl.DataFrame({"x": [1.5, 2.5, 3.5, 4.5, 5.5]})
        result = diff_datasets(orig, repl, privacy=True)

        assert result.privacy is not None
        assert "min" in result.privacy
        assert "mean" in result.privacy

    def test_without_privacy(self):
        orig = pl.DataFrame({"x": [1.0, 2.0]})
        repl = pl.DataFrame({"x": [1.5, 2.5]})
        result = diff_datasets(orig, repl, privacy=False)

        assert result.privacy is None


class TestDiffSharedColumns:
    def test_only_shared_columns(self):
        orig = pl.DataFrame({"x": [1.0], "y": [2.0]})
        repl = pl.DataFrame({"x": [1.5], "z": [3.0]})
        result = diff_datasets(orig, repl)

        assert len(result.column_diffs) == 1
        assert result.column_diffs[0].name == "x"
