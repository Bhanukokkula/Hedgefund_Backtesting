"""Tests for the multiple-testing-aware metrics: Sharpe, Deflated Sharpe
Ratio, and the Bonferroni haircut Sharpe. These check directional/
monotonicity properties (more trials -> more skepticism) rather than exact
numeric matches to literature tables, since the haircut implementation is
an explicitly simplified approximation (see module docstring)."""

from __future__ import annotations

import numpy as np

from glassbox.validation.metrics import (
    deflated_sharpe_ratio,
    harvey_liu_haircut_sharpe,
    sharpe_ratio,
)


def test_sharpe_ratio_zero_for_constant_returns():
    returns = np.zeros(252)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_known_value():
    rng = np.random.default_rng(0)
    daily_mean, daily_std = 0.001, 0.01
    returns = rng.normal(daily_mean, daily_std, 5000)
    sr = sharpe_ratio(returns)
    expected = daily_mean / daily_std * np.sqrt(252)
    assert abs(sr - expected) < 0.5  # sampling noise tolerance


def test_deflated_sharpe_decreases_with_more_trials():
    """The same observed Sharpe should look less credible the more configs
    were tried to find it — DSR (a probability) must fall as n_trials rises."""
    dsr_1 = deflated_sharpe_ratio(observed_sharpe=0.3, n_trials=1, n_obs=252)
    dsr_10 = deflated_sharpe_ratio(observed_sharpe=0.3, n_trials=10, n_obs=252)
    dsr_100 = deflated_sharpe_ratio(observed_sharpe=0.3, n_trials=100, n_obs=252)
    assert dsr_1 > dsr_10 > dsr_100


def test_deflated_sharpe_is_a_probability():
    dsr = deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=5, n_obs=500)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_uses_empirical_trial_variance_when_given():
    trial_sharpes = np.array([0.1, 0.2, -0.1, 0.05, 1.0])
    dsr_empirical = deflated_sharpe_ratio(
        observed_sharpe=1.0, n_trials=5, n_obs=252, trial_sharpes=trial_sharpes
    )
    dsr_fallback = deflated_sharpe_ratio(observed_sharpe=1.0, n_trials=5, n_obs=252)
    assert dsr_empirical != dsr_fallback


def test_haircut_sharpe_never_exceeds_observed_sharpe_in_magnitude():
    observed = 1.2
    haircut = harvey_liu_haircut_sharpe(observed, n_trials=20, n_obs=500)
    assert abs(haircut) <= abs(observed) + 1e-9


def test_haircut_sharpe_decreases_with_more_trials():
    haircut_few = harvey_liu_haircut_sharpe(0.3, n_trials=2, n_obs=500)
    haircut_many = harvey_liu_haircut_sharpe(0.3, n_trials=200, n_obs=500)
    assert haircut_many < haircut_few


def test_haircut_sharpe_single_trial_equals_observed():
    assert harvey_liu_haircut_sharpe(0.8, n_trials=1, n_obs=252) == 0.8


def test_haircut_sharpe_can_collapse_to_zero_for_weak_signal_many_trials():
    haircut = harvey_liu_haircut_sharpe(0.1, n_trials=10_000, n_obs=100)
    assert haircut == 0.0


def test_dsr_with_annualized_sharpe_and_daily_n_obs_saturates_uninformatively():
    """Regression test documenting the exact bug this caught: feeding an
    ANNUALIZED Sharpe alongside a DAILY observation count inflates
    sqrt(n_obs-1) by ~100x, pinning DSR near 1.0 for both a strong and a
    weak factor — destroying the metric's ability to discriminate. This is
    the wrong way to call the function; it's tested here so the failure
    mode stays documented and doesn't silently reappear at a call site."""
    n_obs_daily = 11_000  # ~47 years of daily returns, the real M5 sample
    strong_annualized_sharpe = 0.35
    weak_annualized_sharpe = 0.04
    dsr_strong = deflated_sharpe_ratio(strong_annualized_sharpe, n_trials=3, n_obs=n_obs_daily)
    dsr_weak = deflated_sharpe_ratio(weak_annualized_sharpe, n_trials=3, n_obs=n_obs_daily)
    assert dsr_strong > 0.999
    assert dsr_weak > 0.999  # both saturate — this is the bug, not a feature


def test_dsr_with_consistent_daily_units_discriminates_strong_from_weak():
    """The correct call: convert the annualized Sharpe to a per-day Sharpe
    (divide by sqrt(252)) before pairing it with the daily n_obs. With
    consistent units, a strong factor and a weak one must NOT produce the
    same (saturated) DSR."""
    n_obs_daily = 11_000
    strong_daily_sharpe = 0.35 / np.sqrt(252)
    weak_daily_sharpe = 0.04 / np.sqrt(252)
    dsr_strong = deflated_sharpe_ratio(strong_daily_sharpe, n_trials=3, n_obs=n_obs_daily)
    dsr_weak = deflated_sharpe_ratio(weak_daily_sharpe, n_trials=3, n_obs=n_obs_daily)
    assert dsr_strong > dsr_weak
    assert dsr_weak < 0.9  # a 0.04 annualized Sharpe should NOT read as "almost certainly real"
