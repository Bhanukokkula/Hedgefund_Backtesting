"""The look-ahead audit: deliberately inject a known look-ahead leak and
measure what it's worth.

Two separate claims are being proven here, not one:
  1. The AsOfAccessor itself refuses to serve future data when used
     correctly — already proven by tests/test_asof_adversarial.py. Nothing
     in this module re-implements that; it is referenced, not re-tested.
  2. If a strategy bypasses the accessor's as_of binding (the only way to
     "leak" — there is no other code path into a leak), the resulting
     performance improvement is exactly quantifiable. That's what
     `run_lookahead_audit` measures: it builds two otherwise-identical
     rebalance schedules, one using only data knowable at each signal date
     and one deliberately computed `leak_days` ahead, and runs both through
     the same BacktestEngine so the only difference is the leak.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from glassbox.engine.asof_accessor import ConcreteAsOfAccessor
from glassbox.engine.backtest import BacktestEngine
from glassbox.engine.costs import CostModel
from glassbox.factors.ranking import long_only_top_decile_weights
from glassbox.validation.metrics import sharpe_ratio

ScoreFn = Callable[[ConcreteAsOfAccessor, list[str]], dict[str, float]]


@dataclass(frozen=True)
class LookaheadAuditResult:
    clean_sharpe: float
    leaky_sharpe: float
    clean_final_nav: float
    leaky_final_nav: float
    sharpe_improvement_from_leak: float
    nav_improvement_from_leak: float


def _build_rebalance_schedule(
    panel: dict[str, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp],
    tickers: list[str],
    score_fn: ScoreFn,
    n_deciles: int,
    leak_days: int,
) -> dict[pd.Timestamp, dict[str, float]]:
    schedule = {}
    for as_of in rebalance_dates:
        effective_date = as_of + pd.Timedelta(days=leak_days) if leak_days else as_of
        accessor = ConcreteAsOfAccessor(as_of_date=effective_date.date(), panel=panel)
        scores = score_fn(accessor, tickers)
        weights = long_only_top_decile_weights(scores, n_deciles=n_deciles)
        if weights:
            schedule[as_of] = weights
    return schedule


def run_lookahead_audit(
    panel: dict[str, pd.DataFrame],
    trading_dates: list[pd.Timestamp],
    rebalance_dates: list[pd.Timestamp],
    tickers: list[str],
    score_fn: ScoreFn,
    cost_model: CostModel,
    initial_cash: float,
    leak_days: int,
    n_deciles: int = 10,
) -> LookaheadAuditResult:
    clean_schedule = _build_rebalance_schedule(
        panel, rebalance_dates, tickers, score_fn, n_deciles, leak_days=0
    )
    leaky_schedule = _build_rebalance_schedule(
        panel, rebalance_dates, tickers, score_fn, n_deciles, leak_days=leak_days
    )

    clean_engine = BacktestEngine(panel, trading_dates, clean_schedule, cost_model, initial_cash)
    leaky_engine = BacktestEngine(panel, trading_dates, leaky_schedule, cost_model, initial_cash)

    clean_history = clean_engine.run()
    leaky_history = leaky_engine.run()

    clean_navs = pd.Series([r.nav for r in clean_history])
    leaky_navs = pd.Series([r.nav for r in leaky_history])
    clean_returns = clean_navs.pct_change().dropna().to_numpy()
    leaky_returns = leaky_navs.pct_change().dropna().to_numpy()

    clean_sharpe = sharpe_ratio(clean_returns)
    leaky_sharpe = sharpe_ratio(leaky_returns)

    return LookaheadAuditResult(
        clean_sharpe=clean_sharpe,
        leaky_sharpe=leaky_sharpe,
        clean_final_nav=float(clean_navs.iloc[-1]),
        leaky_final_nav=float(leaky_navs.iloc[-1]),
        sharpe_improvement_from_leak=leaky_sharpe - clean_sharpe,
        nav_improvement_from_leak=float(leaky_navs.iloc[-1] - clean_navs.iloc[-1]),
    )
