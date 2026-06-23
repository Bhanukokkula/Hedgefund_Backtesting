"""M5 acceptance test: the look-ahead audit must demonstrably catch the
injected leak — i.e. show a measurable, positive improvement when a
strategy is allowed to peek `leak_days` into the future relative to the
honest as-of view.

Deterministic construction (no randomness): T_JUMP re-rates sharply on a
known day; T_FLAT never moves. The signal date is 5 trading days before the
jump. With no leak, both names look identical (flat, 0% trailing return) at
the signal date, and a deterministic tie-break picks T_FLAT. With a leak
large enough to see past the jump, T_JUMP's trailing return turns positive
and the leaky schedule rotates into it — at the SAME signal date, well
before the jump happens in the engine's own timeline — capturing a gain the
honest version cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from glassbox.engine.costs import CostModel
from glassbox.validation.lookahead_audit import run_lookahead_audit

ZERO_COSTS = CostModel(
    commission_bps=0.0,
    half_spread_bps=0.0,
    market_impact_coefficient=0.0,
    participation_rate_cap=1.0,
)

N_DAYS = 60
JUMP_AT = 30


def _make_df(dates, close):
    return pd.DataFrame(
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


def _build_panel():
    dates = pd.bdate_range("2019-01-01", periods=N_DAYS)
    close_jump = np.full(N_DAYS, 100.0)
    close_jump[JUMP_AT:] *= 1.5
    close_flat = np.full(N_DAYS, 100.0)
    panel = {"T_JUMP": _make_df(dates, close_jump), "T_FLAT": _make_df(dates, close_flat)}
    return panel, dates


def _five_day_return_score(accessor, tickers):
    scores = {}
    for ticker in tickers:
        series = accessor.price_series(ticker, lookback_days=5, adjusted=True)
        if len(series) < 2:
            scores[ticker] = 0.0
            continue
        scores[ticker] = float(series.iloc[-1] / series.iloc[0] - 1.0)
    return scores


def test_lookahead_audit_shows_positive_improvement_from_injected_leak():
    panel, dates = _build_panel()
    tickers = ["T_JUMP", "T_FLAT"]
    rebalance_dates = [dates[JUMP_AT - 5]]

    result = run_lookahead_audit(
        panel=panel,
        trading_dates=list(dates),
        rebalance_dates=rebalance_dates,
        tickers=tickers,
        score_fn=_five_day_return_score,
        cost_model=ZERO_COSTS,
        initial_cash=10_000.0,
        leak_days=10,
        n_deciles=2,
    )

    assert result.nav_improvement_from_leak > 0
    assert result.leaky_final_nav > result.clean_final_nav
    # the clean run never saw the jump coming, so it holds the flat name
    assert abs(result.clean_final_nav - 10_000.0) < 1e-6
    # the leaky run captured the 50% re-rating
    assert result.leaky_final_nav > 14_000.0


def test_lookahead_audit_with_zero_leak_days_gives_identical_results():
    panel, dates = _build_panel()
    tickers = ["T_JUMP", "T_FLAT"]
    rebalance_dates = [dates[JUMP_AT - 5]]

    result = run_lookahead_audit(
        panel=panel,
        trading_dates=list(dates),
        rebalance_dates=rebalance_dates,
        tickers=tickers,
        score_fn=_five_day_return_score,
        cost_model=ZERO_COSTS,
        initial_cash=10_000.0,
        leak_days=0,
        n_deciles=2,
    )
    assert abs(result.nav_improvement_from_leak) < 1e-9
    assert result.clean_final_nav == result.leaky_final_nav
