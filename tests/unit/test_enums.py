"""Tests for profile schema enum definitions."""

from proof_of_replica.core.enums import (
    ColumnDtype,
    ColumnRole,
    CorrelationMethod,
    DistributionFamily,
    GeneratorMethod,
    MissingnessPattern,
)


class TestColumnDtype:
    def test_has_expected_members(self):
        assert set(ColumnDtype) == {
            ColumnDtype.STRING,
            ColumnDtype.FLOAT64,
            ColumnDtype.INT64,
            ColumnDtype.CATEGORICAL,
            ColumnDtype.BOOLEAN,
            ColumnDtype.DATE,
        }

    def test_values_are_lowercase_strings(self):
        for member in ColumnDtype:
            assert member.value == member.value.lower()
            assert isinstance(member.value, str)


class TestColumnRole:
    def test_has_expected_members(self):
        assert set(ColumnRole) == {
            ColumnRole.IDENTIFIER,
            ColumnRole.FEATURE,
            ColumnRole.GROUP,
            ColumnRole.OUTCOME,
            ColumnRole.INDEX,
        }


class TestDistributionFamily:
    def test_has_expected_members(self):
        expected = {
            "normal",
            "lognormal",
            "uniform",
            "beta",
            "gamma",
            "exponential",
            "empirical_kde",
            "empirical_histogram",
        }
        assert {m.value for m in DistributionFamily} == expected


class TestGeneratorMethod:
    def test_has_expected_members(self):
        expected = {
            "sequential",
            "uuid",
            "prefix_increment",
            "regex",
            "alphabet",
            "weighted_choice",
            "placeholder",
            "lorem",
            "template",
            "grammar",
        }
        assert {m.value for m in GeneratorMethod} == expected


class TestCorrelationMethod:
    def test_has_expected_members(self):
        assert set(CorrelationMethod) == {
            CorrelationMethod.PEARSON,
            CorrelationMethod.SPEARMAN,
            CorrelationMethod.KENDALL,
        }


class TestMissingnessPattern:
    def test_has_expected_members(self):
        expected = {"MCAR", "MAR", "MNAR", "observed"}
        assert {m.value for m in MissingnessPattern} == expected

    def test_observed_is_lowercase(self):
        assert MissingnessPattern.OBSERVED.value == "observed"
