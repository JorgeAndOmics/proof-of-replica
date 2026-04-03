"""Tests for distribution fitting engine."""

import numpy as np
import pytest

from proof_of_replica.core.enums import DistributionFamily
from proof_of_replica.core.schema import ProfilerConfig
from proof_of_replica.engines._distribution_fit import fit_distribution


@pytest.fixture
def config() -> ProfilerConfig:
    return ProfilerConfig()


class TestFitDistribution:
    def test_fits_normal(self, config):
        rng = np.random.default_rng(42)
        values = rng.normal(50, 10, size=5000)
        result = fit_distribution(values, config)
        assert result.family == DistributionFamily.NORMAL
        assert abs(result.params["loc"] - 50) < 2  # type: ignore[operator]
        assert abs(result.params["scale"] - 10) < 2  # type: ignore[operator]

    def test_fits_exponential(self, config):
        rng = np.random.default_rng(42)
        values = rng.exponential(scale=5.0, size=5000)
        result = fit_distribution(values, config)
        # Should pick exponential or gamma (exponential is gamma with a=1)
        assert result.family in (
            DistributionFamily.EXPONENTIAL,
            DistributionFamily.GAMMA,
        )

    def test_fits_uniform(self, config):
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 10, size=5000)
        result = fit_distribution(values, config)
        assert result.family == DistributionFamily.UNIFORM

    def test_fallback_to_kde_small_sample(self, config):
        values = np.array([1.0, 2.0, 3.0])
        result = fit_distribution(values, config)
        assert result.family == DistributionFamily.EMPIRICAL_KDE

    def test_fallback_to_kde_no_good_fit(self):
        # Very strict threshold should cause fallback
        config = ProfilerConfig(distribution_fit_pvalue=0.99)
        rng = np.random.default_rng(42)
        # Bimodal data that no single parametric family fits well
        values = np.concatenate(
            [
                rng.normal(-10, 1, size=500),
                rng.normal(10, 1, size=500),
            ]
        )
        result = fit_distribution(values, config)
        assert result.family == DistributionFamily.EMPIRICAL_KDE

    def test_skips_lognormal_for_negative_data(self, config):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=5000)  # Has negative values
        result = fit_distribution(values, config)
        # Should not pick lognormal
        assert result.family != DistributionFamily.LOGNORMAL

    def test_skips_beta_for_out_of_range(self, config):
        rng = np.random.default_rng(42)
        values = rng.normal(50, 10, size=5000)  # Not in [0, 1]
        result = fit_distribution(values, config)
        assert result.family != DistributionFamily.BETA

    def test_empty_values_fallback(self, config):
        values = np.array([], dtype=np.float64)
        result = fit_distribution(values, config)
        assert result.family == DistributionFamily.EMPIRICAL_KDE

    def test_deterministic(self, config):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=1000)
        r1 = fit_distribution(values, config)
        r2 = fit_distribution(values, config)
        assert r1.family == r2.family
        assert r1.params == r2.params
