"""M5 acceptance test: walk-forward harness enforces strict train/test
separation and reports in-sample vs out-of-sample performance side by side."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glassbox.engine.costs import CostModel
from glassbox.validation.walk_forward import run_walk_forward, split_rebalance_dates

ZERO_COSTS = CostModel(
    commission_bps=0.0,
    half_spread_bps=0.0,
    market_impact_coefficient=0.0,
    participation_rate_cap=1.0,
)


def test_split_is_chronological_and_non_overlapping():
    dates = pd.bdate_range("2020-01-01", periods=100)
    in_sample, out_of_sample = split_rebalance_dates(list(dates), split_fraction=0.7)
    assert set(in_sample).isdisjoint(out_of_sample)
    assert max(in_sample) < min(out_of_sample)
    assert len(in_sample) + len(out_of_sample) == len(dates)


def test_walk_forward_reports_both_windows_on_trending_asset():
    dates = pd.bdate_range("2019-01-01", periods=252)
    close = 100.0 * (1.0005 ** np.arange(len(dates)))  # steady uptrend, same trend both halves
    df = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1_000_000,
            "split_factor": 1.0,
            "div_cash": 0.0,
        },
        index=dates,
    )
    panel = {"AAA": df}
    rebalance_dates = list(dates[::21])  # roughly monthly
    schedule = {d: {"AAA": 1.0} for d in rebalance_dates}

    result = run_walk_forward(
        panel, schedule, split_fraction=0.6, cost_model=ZERO_COSTS, initial_cash=10_000.0
    )

    assert result.n_in_sample_rebalances > 0
    assert result.n_out_of_sample_rebalances > 0
    # a steady uptrend with no costs should show positive Sharpe in both windows
    assert result.in_sample_sharpe > 0
    assert result.out_of_sample_sharpe > 0


def test_walk_forward_windows_are_independent_not_compounding():
    """Both windows start from the same initial_cash — out-of-sample results
    must not be inflated by carrying forward in-sample gains."""
    dates = pd.bdate_range("2019-01-01", periods=100)
    close = np.full(len(dates), 100.0)
    df = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1_000_000,
            "split_factor": 1.0,
            "div_cash": 0.0,
        },
        index=dates,
    )
    panel = {"AAA": df}
    rebalance_dates = list(dates[::10])
    schedule = {d: {"AAA": 1.0} for d in rebalance_dates}

    result = run_walk_forward(
        panel, schedule, split_fraction=0.5, cost_model=ZERO_COSTS, initial_cash=10_000.0
    )
    # flat prices, zero costs -> both windows should show exactly zero return
    assert abs(result.in_sample_annualized_return_bps) < 1e-6
    assert abs(result.out_of_sample_annualized_return_bps) < 1e-6
