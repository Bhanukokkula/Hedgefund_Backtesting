"""Walk-forward / out-of-sample harness: strict train/test separation.

There is no parameter-fitting step in this project's factors (momentum,
reversal, low-vol all use fixed lookback windows from config.yaml, not
fitted ones), so "in-sample vs out-of-sample" here means: does the SAME
factor definition keep working on a later period it was never looked at
while being designed? That's the discipline this module enforces — the
in-sample and out-of-sample backtests never share a rebalance date, and
each runs as an independent BacktestEngine starting from the same initial
cash, so one period's results cannot leak into or compound with the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from glassbox.engine.backtest import BacktestEngine
from glassbox.engine.costs import CostModel
from glassbox.validation.metrics import sharpe_ratio


@dataclass(frozen=True)
class WalkForwardResult:
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_annualized_return_bps: float
    out_of_sample_annualized_return_bps: float
    n_in_sample_rebalances: int
    n_out_of_sample_rebalances: int


def split_rebalance_dates(
    rebalance_dates: list[pd.Timestamp], split_fraction: float = 0.7
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Chronological split — no shuffling, no random split: in-sample is
    strictly everything before the cut, out-of-sample strictly after."""
    idx = int(len(rebalance_dates) * split_fraction)
    return rebalance_dates[:idx], rebalance_dates[idx:]


def _run_window(
    panel: dict[str, pd.DataFrame],
    trading_dates: list[pd.Timestamp],
    rebalance_schedule: dict[pd.Timestamp, dict[str, float]],
    cost_model: CostModel,
    initial_cash: float,
) -> tuple[float, float]:
    engine = BacktestEngine(panel, trading_dates, rebalance_schedule, cost_model, initial_cash)
    history = engine.run()
    navs = pd.Series([r.nav for r in history])
    returns = navs.pct_change().dropna().to_numpy()
    sharpe = sharpe_ratio(returns)
    ann_return_bps = float(returns.mean() * 252 * 10_000) if len(returns) else 0.0
    return sharpe, ann_return_bps


def run_walk_forward(
    panel: dict[str, pd.DataFrame],
    full_rebalance_schedule: dict[pd.Timestamp, dict[str, float]],
    split_fraction: float,
    cost_model: CostModel,
    initial_cash: float,
) -> WalkForwardResult:
    rebalance_dates = sorted(full_rebalance_schedule)
    in_sample_dates, out_of_sample_dates = split_rebalance_dates(rebalance_dates, split_fraction)

    in_sample_schedule = {d: full_rebalance_schedule[d] for d in in_sample_dates}
    out_of_sample_schedule = {d: full_rebalance_schedule[d] for d in out_of_sample_dates}

    all_dates = sorted({ts for df in panel.values() for ts in df.index})
    in_sample_trading_dates = (
        [d for d in all_dates if d <= in_sample_dates[-1]] if in_sample_dates else []
    )
    out_of_sample_trading_dates = (
        [d for d in all_dates if d >= out_of_sample_dates[0]] if out_of_sample_dates else []
    )

    in_sharpe, in_ann_bps = _run_window(
        panel, in_sample_trading_dates, in_sample_schedule, cost_model, initial_cash
    )
    out_sharpe, out_ann_bps = _run_window(
        panel, out_of_sample_trading_dates, out_of_sample_schedule, cost_model, initial_cash
    )

    return WalkForwardResult(
        in_sample_sharpe=in_sharpe,
        out_of_sample_sharpe=out_sharpe,
        in_sample_annualized_return_bps=in_ann_bps,
        out_of_sample_annualized_return_bps=out_ann_bps,
        n_in_sample_rebalances=len(in_sample_dates),
        n_out_of_sample_rebalances=len(out_of_sample_dates),
    )
