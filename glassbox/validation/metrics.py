"""Multiple-testing-aware performance metrics: the Deflated Sharpe Ratio and
a Harvey-Liu-style haircut Sharpe, alongside the naive Sharpe. These exist
because trying several factors/parameter configurations and reporting only
the best one's Sharpe ratio is itself a form of lying — the naive Sharpe
overstates how likely the result is genuine skill rather than the best of N
noisy draws.

Deflated Sharpe Ratio (DSR): Bailey & López de Prado (2014), "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and
Non-Normality". Returns P(true Sharpe > 0 | observed Sharpe, n_trials,
sample skew/kurtosis) — NOT a Sharpe ratio itself, a probability in [0, 1].

Haircut Sharpe: a SIMPLIFIED, Bonferroni-based approximation in the spirit
of Harvey & Liu (2015), "Backtesting" — not a reproduction of their full
regression-based haircut tables (those require empirical inputs this
project does not have). Documented here as an approximation, not a claim of
exact replication.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def _expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max of n_trials iid Sharpe-ratio estimates with variance sr_variance],
    using the Bailey & López de Prado closed-form approximation based on
    the expected maximum of n iid standard normals."""
    if n_trials <= 1:
        return 0.0
    return float(
        np.sqrt(sr_variance)
        * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    trial_sharpes: np.ndarray | None = None,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """P(true Sharpe > 0), correcting for having tried `n_trials` configs.

    `observed_sharpe` MUST be the Sharpe computed at the SAME frequency as
    `n_obs` counts observations — e.g. a per-day Sharpe (no annualization
    factor) with n_obs = number of daily returns, or a per-month Sharpe
    with n_obs = number of months. Passing an annualized Sharpe (which
    `sharpe_ratio()` returns by default, scaled by sqrt(252)) together with
    a daily n_obs inflates sqrt(n_obs-1) by ~100x relative to the Sharpe's
    own scale, saturating the result to ~1.0 regardless of the factor's
    actual quality — get the per-period Sharpe via
    `sharpe_ratio(returns, periods_per_year=1)`.

    If `trial_sharpes` (the Sharpe ratios of all trials actually run) is
    given, its sample variance estimates the variance of the Sharpe-ratio
    estimator directly. Otherwise falls back to the standard-error
    approximation Var[SR] ~= 1/n_obs.
    """
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_variance = float(np.var(trial_sharpes, ddof=1))
    else:
        sr_variance = 1.0 / n_obs

    sr0 = _expected_max_sharpe(n_trials, sr_variance)
    denom = np.sqrt(
        max(
            1e-12,
            1 - skewness * observed_sharpe + (excess_kurtosis / 4) * observed_sharpe**2,
        )
    )
    dsr_stat = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(dsr_stat))


def harvey_liu_haircut_sharpe(observed_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Bonferroni-adjusted haircut Sharpe: shrinks `observed_sharpe` toward
    zero by the same ratio the significance threshold must tighten to
    account for `n_trials` independent attempts. A simplified stand-in for
    Harvey & Liu's full regression-based haircut model.

    Same unit requirement as `deflated_sharpe_ratio`: `observed_sharpe` and
    `n_obs` must be at the same frequency (e.g. both daily). The returned
    value is at that same frequency too — re-annualize it yourself
    (multiply by sqrt(periods_per_year)) if you need it next to an
    annualized naive Sharpe for display.

    Implemented as a ratio of z-scores (post- vs pre-Bonferroni-adjustment)
    applied multiplicatively to `observed_sharpe`, rather than recomputing
    an unrelated threshold-implied Sharpe — this keeps the result bounded
    by construction (the ratio is always <= 1), avoiding the inf/overflow
    failure mode of inverse-mapping a near-zero adjusted p-value directly.
    """
    if n_trials <= 1 or n_obs <= 1 or observed_sharpe == 0:
        return observed_sharpe

    z_original = abs(observed_sharpe) * np.sqrt(n_obs)
    p_value_single = 2 * (1 - norm.cdf(z_original))
    p_value_adjusted = min(1.0, p_value_single * n_trials)

    if p_value_adjusted >= 1.0:
        return 0.0

    z_adjusted = norm.ppf(1 - p_value_adjusted / 2)
    ratio = min(1.0, z_adjusted / z_original) if z_original > 0 else 0.0
    sign = 1.0 if observed_sharpe >= 0 else -1.0
    return float(sign * abs(observed_sharpe) * ratio)
