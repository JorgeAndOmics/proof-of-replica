"""Distribution fitting via AIC selection for profiling."""

import logging
import warnings

import numpy as np
import scipy.stats

from proof_of_replica.core.column_schema import DistributionConfig
from proof_of_replica.core.enums import DistributionFamily
from proof_of_replica.core.schema import ProfilerConfig

logger = logging.getLogger(__name__)

# Candidate families and their scipy counterparts + param count
_CANDIDATES: list[tuple[DistributionFamily, type[scipy.stats.rv_continuous], int]] = [
    (DistributionFamily.NORMAL, scipy.stats.norm, 2),
    (DistributionFamily.LOGNORMAL, scipy.stats.lognorm, 3),
    (DistributionFamily.GAMMA, scipy.stats.gamma, 3),
    (DistributionFamily.BETA, scipy.stats.beta, 4),
    (DistributionFamily.UNIFORM, scipy.stats.uniform, 2),
    (DistributionFamily.EXPONENTIAL, scipy.stats.expon, 2),
]


def fit_distribution(
    values: np.ndarray,
    config: ProfilerConfig,
) -> DistributionConfig:
    """Fit the best parametric distribution to continuous data.

    Tries candidate distributions, selects by lowest AIC. Falls back
    to empirical KDE if no parametric fit passes the goodness-of-fit
    threshold.

    Args:
        values: 1D array of non-null numeric values.
        config: Profiler configuration with fit thresholds.

    Returns:
        DistributionConfig with the best-fit family and parameters.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 5:
        return _fallback_kde(clean)

    best_aic = float("inf")
    best_family: DistributionFamily | None = None
    best_params: dict[str, float | str] = {}

    for family, dist_cls, k in _CANDIDATES:
        if not _is_eligible(clean, family):
            continue

        result = _try_fit(clean, family, dist_cls, k)
        if result is not None and result[1] < best_aic:
            best_family, best_aic, best_params = family, result[1], result[0]

    if best_family is None:
        return _fallback_kde(clean)

    # Goodness-of-fit check via KS test
    dist_cls_best = next(c for f, c, _ in _CANDIDATES if f == best_family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value = scipy.stats.kstest(
            clean,
            dist_cls_best(**{k: float(v) for k, v in best_params.items()}).cdf,
        )

    if p_value < config.distribution_fit_pvalue:
        return _fallback_kde(clean)

    return DistributionConfig(family=best_family, params=best_params)


def _is_eligible(values: np.ndarray, family: DistributionFamily) -> bool:
    """Check if data is eligible for a given distribution family."""
    if family == DistributionFamily.LOGNORMAL and np.any(values <= 0):
        return False
    if family == DistributionFamily.GAMMA and np.any(values <= 0):
        return False
    if family == DistributionFamily.BETA:
        return bool(np.all(values >= 0) and np.all(values <= 1))
    if family == DistributionFamily.EXPONENTIAL:
        return bool(np.all(values >= 0))
    return True


def _try_fit(
    values: np.ndarray,
    family: DistributionFamily,
    dist_cls: type[scipy.stats.rv_continuous],
    k: int,
) -> tuple[dict[str, float | str], float] | None:
    """Try fitting a distribution, returning (params, aic) or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_params = dist_cls.fit(values)

        dist = dist_cls(*fit_params)
        log_lik = np.sum(dist.logpdf(values))

        if not np.isfinite(log_lik):
            return None

        aic = 2.0 * k - 2.0 * log_lik
        params = _scipy_params_to_dict(family, fit_params)
        return params, aic
    except Exception:
        return None


_PARAM_NAMES: dict[DistributionFamily, tuple[str, ...]] = {
    DistributionFamily.NORMAL: ("loc", "scale"),
    DistributionFamily.LOGNORMAL: ("s", "loc", "scale"),
    DistributionFamily.GAMMA: ("a", "loc", "scale"),
    DistributionFamily.BETA: ("a", "b", "loc", "scale"),
    DistributionFamily.UNIFORM: ("loc", "scale"),
    DistributionFamily.EXPONENTIAL: ("loc", "scale"),
}


def _scipy_params_to_dict(
    family: DistributionFamily,
    fit_params: tuple[float, ...],
) -> dict[str, float | str]:
    """Convert scipy fit parameters to the profile's params dict format."""
    names = _PARAM_NAMES.get(family, ("loc", "scale"))
    return dict(zip(names, fit_params, strict=False))


def _fallback_kde(values: np.ndarray) -> DistributionConfig:
    """Fall back to empirical KDE when no parametric fit is adequate."""
    if len(values) == 0:
        return DistributionConfig(
            family=DistributionFamily.EMPIRICAL_KDE,
            params={"bandwidth": 1.0},
        )

    std = float(np.std(values))
    n = len(values)
    # Scott's rule for bandwidth estimation
    bandwidth = 1.06 * std * n ** (-1.0 / 5.0) if std > 0 else 1.0

    return DistributionConfig(
        family=DistributionFamily.EMPIRICAL_KDE,
        params={"bandwidth": bandwidth},
    )
