"""Tests for column-level schema models."""

import pytest
from pydantic import ValidationError

from proof_of_replica.core.column_schema import (
    BooleanStats,
    CategoricalStats,
    ColumnConstraints,
    ColumnOverrides,
    DateStats,
    DistributionConfig,
    GroupEffect,
    GroupEffects,
    NumericStats,
    StringStats,
    ValidationOverrides,
)
from proof_of_replica.core.enums import DistributionFamily

# ── NumericStats ──────────────────────────────────────────────


class TestNumericStats:
    def test_construction_with_all_fields(self):
        stats = NumericStats(
            mean=52.3,
            std=14.7,
            min=18.0,
            max=89.0,
            median=51.0,
            skewness=0.23,
            kurtosis=-0.41,
            null_fraction=0.02,
            percentiles={"5": 28.1, "25": 42.0, "75": 62.5, "95": 76.8},
        )
        assert stats.mean == 52.3
        assert stats.percentiles == {"5": 28.1, "25": 42.0, "75": 62.5, "95": 76.8}

    def test_minimal_construction(self):
        stats = NumericStats(mean=50.0, std=10.0)
        assert stats.null_fraction == 0.0
        assert stats.percentiles is None

    def test_is_frozen(self):
        stats = NumericStats(mean=50.0, std=10.0)
        with pytest.raises(ValidationError):
            stats.mean = 99.0  # type: ignore[misc]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            NumericStats(mean=50.0, std=10.0, bogus=1)  # type: ignore[call-arg]

    def test_null_fraction_range(self):
        with pytest.raises(ValidationError):
            NumericStats(mean=50.0, std=10.0, null_fraction=1.5)

    def test_json_round_trip(self):
        stats = NumericStats(mean=52.3, std=14.7, min=18.0, max=89.0)
        data = stats.model_dump()
        restored = NumericStats.model_validate(data)
        assert restored == stats


# ── CategoricalStats ──────────────────────────────────────────


class TestCategoricalStats:
    def test_construction(self):
        stats = CategoricalStats(
            cardinality=3,
            value_counts={"a": 0.4, "b": 0.35, "c": 0.25},
        )
        assert stats.cardinality == 3
        assert stats.null_fraction == 0.0

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            CategoricalStats(
                cardinality=3,
                value_counts={"a": 1.0},
                mean=5.0,  # type: ignore[call-arg]
            )


# ── BooleanStats ──────────────────────────────────────────────


class TestBooleanStats:
    def test_construction(self):
        stats = BooleanStats(true_fraction=0.4)
        assert stats.true_fraction == 0.4
        assert stats.null_fraction == 0.0

    def test_true_fraction_range(self):
        with pytest.raises(ValidationError):
            BooleanStats(true_fraction=1.5)
        with pytest.raises(ValidationError):
            BooleanStats(true_fraction=-0.1)


# ── DateStats ──────────────────────────────────────────────


class TestDateStats:
    def test_construction(self):
        stats = DateStats(min="2020-01-15", max="2025-11-30", null_fraction=0.05)
        assert stats.min == "2020-01-15"

    def test_minimal(self):
        stats = DateStats()
        assert stats.null_fraction == 0.0


# ── StringStats ──────────────────────────────────────────────


class TestStringStats:
    def test_construction(self):
        stats = StringStats(null_fraction=0.6, mean_length=42.0, max_length=256)
        assert stats.mean_length == 42.0


# ── DistributionConfig ──────────────────────────────────────────


class TestDistributionConfig:
    def test_normal_distribution(self):
        dist = DistributionConfig(
            family=DistributionFamily.NORMAL,
            params={"loc": 52.3, "scale": 14.7},
        )
        assert dist.family == "normal"
        assert dist.params["loc"] == 52.3

    def test_empty_params_allowed(self):
        dist = DistributionConfig(family=DistributionFamily.UNIFORM)
        assert dist.params == {}

    def test_rejects_unknown_family(self):
        with pytest.raises(ValidationError):
            DistributionConfig(family="bogus_distribution")  # type: ignore[arg-type]

    def test_is_frozen(self):
        dist = DistributionConfig(family=DistributionFamily.NORMAL)
        with pytest.raises(ValidationError):
            dist.family = DistributionFamily.BETA  # type: ignore[misc]

    def test_json_round_trip(self):
        dist = DistributionConfig(
            family=DistributionFamily.LOGNORMAL,
            params={"s": 0.8, "loc": 0.0, "scale": 12.5},
        )
        restored = DistributionConfig.model_validate(dist.model_dump())
        assert restored == dist


# ── ColumnConstraints ──────────────────────────────────────────


class TestColumnConstraints:
    def test_defaults(self):
        c = ColumnConstraints()
        assert c.min is None
        assert c.max is None
        assert c.integer_valued is False

    def test_with_values(self):
        c = ColumnConstraints(min=18.0, max=100.0, integer_valued=True)
        assert c.min == 18.0


# ── GroupEffect / GroupEffects ──────────────────────────────


class TestGroupEffects:
    def test_construction(self):
        effects = GroupEffects(
            group_column="condition",
            effects={
                "healthy": GroupEffect(shift=0.0, scale_factor=1.0),
                "stage_1": GroupEffect(shift=2.3, scale_factor=1.1),
            },
        )
        assert effects.effects["stage_1"].shift == 2.3

    def test_group_effect_defaults(self):
        effect = GroupEffect()
        assert effect.shift == 0.0
        assert effect.scale_factor == 1.0


# ── ColumnOverrides / ValidationOverrides ──────────────────


class TestColumnOverrides:
    def test_noise_level(self):
        o = ColumnOverrides(noise_level=0.1)
        assert o.noise_level == 0.1


class TestValidationOverrides:
    def test_partial_overrides(self):
        v = ValidationOverrides(ks_pvalue=0.1)
        assert v.ks_pvalue == 0.1
        assert v.null_tolerance is None
