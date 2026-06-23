"""M3 acceptance tests: hand-reconciled P&L for buy-and-hold and all-cash,
correct turnover/cost deduction, and proof that fills never happen on the
signal bar's own close."""

from __future__ import annotations

import pandas as pd

from glassbox.engine.backtest import BacktestEngine
from glassbox.engine.costs import CostModel

ZERO_COSTS = CostModel(
    commission_bps=0.0,
    half_spread_bps=0.0,
    market_impact_coefficient=0.0,
    participation_rate_cap=1.0,
)


def _df(dates, open_, close, volume=1_000_000):
    return pd.DataFrame(
        {"open": open_, "close": close, "volume": volume}, index=pd.DatetimeIndex(dates)
    )


def test_buy_and_hold_reconciles_by_hand():
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
    # signal on day0 (close=100); fill at day1's open=101; price drifts to 110 by day9.
    open_ = [100, 101, 102, 103, 104, 105, 106, 107]
    close = [100, 102, 103, 104, 105, 106, 107, 110]
    panel = {"AAA": _df(dates, open_, close)}
    rebalance_schedule = {dates[0]: {"AAA": 1.0}}

    engine = BacktestEngine(
        panel, list(dates), rebalance_schedule, ZERO_COSTS, initial_cash=10_000.0
    )
    history = engine.run()

    # Shares are sized using the signal day's close (the last known price at
    # decision time), then bought at the next day's open — the realistic fill
    # price can differ from the sizing price, so invested value isn't exactly
    # the full $10,000.
    signal_close = close[0]
    fill_price = open_[1]
    shares = 10_000.0 / signal_close
    cash_after_fill = 10_000.0 - shares * fill_price
    expected_final_nav = shares * close[-1] + cash_after_fill

    assert abs(history[-1].nav - expected_final_nav) < 1e-6
    # day0 NAV (before any fill executes) must still be the untouched cash.
    assert history[0].nav == 10_000.0


def test_all_cash_strategy_nav_stays_flat():
    dates = pd.date_range("2020-01-01", "2020-01-05", freq="B")
    panel = {"AAA": _df(dates, [100] * len(dates), [101] * len(dates))}
    engine = BacktestEngine(panel, list(dates), {}, ZERO_COSTS, initial_cash=5_000.0)
    history = engine.run()
    assert all(record.nav == 5_000.0 for record in history)
    assert engine.portfolio.total_cost_paid == 0.0


def test_no_same_bar_fill():
    """Share count is sized using the signal day's close; the question this
    test isolates is which price the actual cash debit uses to execute that
    share count. If fills happened at the signal bar's own close instead of
    the next bar's open, the cash debited would differ. day0's close and
    day1's open are deliberately very different so a same-bar-fill bug is
    caught by a wrong cash balance."""
    dates = pd.date_range("2020-01-01", "2020-01-06", freq="B")
    open_ = [100, 50, 50, 50]  # day1 open crashes to 50
    close = [200, 50, 50, 50]  # day0 close is 200 — very different from day1 open
    panel = {"AAA": _df(dates, open_, close)}
    rebalance_schedule = {dates[0]: {"AAA": 1.0}}

    engine = BacktestEngine(
        panel, list(dates), rebalance_schedule, ZERO_COSTS, initial_cash=10_000.0
    )
    engine.run()

    shares = 10_000.0 / close[0]  # sized using signal day's close (200)
    cash_if_correct = 10_000.0 - shares * open_[1]  # fills at next bar's open (50)
    cash_if_buggy = 10_000.0 - shares * close[0]  # would fill at signal bar's own close (200)

    assert abs(engine.portfolio.positions["AAA"] - shares) < 1e-6
    assert abs(engine.portfolio.cash - cash_if_correct) < 1e-6
    assert abs(engine.portfolio.cash - cash_if_buggy) > 1.0


def test_turnover_and_costs_deducted_correctly():
    dates = pd.date_range("2020-01-01", "2020-01-06", freq="B")
    panel = {"AAA": _df(dates, [100, 100, 100, 100], [100, 100, 100, 100], volume=1_000_000)}
    cost_model = CostModel(
        commission_bps=10.0,  # 0.10%
        half_spread_bps=5.0,  # 0.05%
        market_impact_coefficient=0.0,
        participation_rate_cap=1.0,
    )
    rebalance_schedule = {dates[0]: {"AAA": 1.0}}
    engine = BacktestEngine(
        panel, list(dates), rebalance_schedule, cost_model, initial_cash=10_000.0
    )
    engine.run()

    fill_price = 100.0
    shares = 10_000.0 / fill_price
    trade_value = shares * fill_price
    expected_cost = trade_value * (10.0 + 5.0) / 10_000

    assert abs(engine.portfolio.total_cost_paid - expected_cost) < 1e-6
    assert abs(engine.portfolio.total_traded_value - trade_value) < 1e-6
    # cash = initial - trade_value - cost
    expected_cash = 10_000.0 - trade_value - expected_cost
    assert abs(engine.portfolio.cash - expected_cash) < 1e-6


def test_rebalance_closes_out_position_not_in_new_target():
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
    panel = {
        "AAA": _df(dates, [100] * len(dates), [100] * len(dates)),
        "BBB": _df(dates, [50] * len(dates), [50] * len(dates)),
    }
    rebalance_schedule = {
        dates[0]: {"AAA": 1.0},
        dates[2]: {"BBB": 1.0},  # rotate fully out of AAA into BBB
    }
    engine = BacktestEngine(
        panel, list(dates), rebalance_schedule, ZERO_COSTS, initial_cash=10_000.0
    )
    engine.run()
    assert "AAA" not in engine.portfolio.positions
    assert "BBB" in engine.portfolio.positions
