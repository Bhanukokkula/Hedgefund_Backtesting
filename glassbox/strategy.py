"""The declarative strategy API: a strategy is data/config, not a script.

This is what makes GLASSBOX a platform rather than a pile of one-off
backtest scripts: `run_strategy(spec, panel, candidates_csv)` is the single
function both the CLI (glassbox.validation.run_m5) and the Streamlit
dashboard (glassbox.reporting.dashboard) call. Same spec, same panel, same
candidates file -> byte-identical numbers, because there's only one
implementation to call, not two that have to be kept in sync by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from glassbox.data.candidates import load_candidate_universe
from glassbox.data.universe import build_survivorship_aware_universe, monthly_rebalance_dates
from glassbox.engine.asof_accessor import ConcreteAsOfAccessor
from glassbox.engine.backtest import BacktestEngine
from glassbox.engine.costs import CostModel
from glassbox.factors.ranking import decile_long_short_weights, long_only_top_decile_weights
from glassbox.factors.scoring import low_vol_score, momentum_score, reversal_score
from glassbox.validation.m1_report import detect_bad_ticks, detect_coverage_gaps
from glassbox.validation.metrics import sharpe_ratio
from glassbox.validation.sensitivity import survivorship_sensitivity

FactorName = Literal["momentum", "reversal", "low_vol"]
ConstructionName = Literal["long_only_top_decile", "decile_long_short"]


class StrategySpec(BaseModel):
    """The entire definition of a strategy: factor + universe + costs +
    construction. Two StrategySpec instances with equal fields are
    guaranteed to produce equal results when run against the same panel."""

    factor: FactorName
    construction: ConstructionName = "long_only_top_decile"
    n_deciles: int = 10
    apply_data_quality_filter: bool = True
    liquidity_top_n: int | None = None
    commission_bps: float = 1.0
    half_spread_bps: float = 5.0
    market_impact_coefficient: float = 0.1
    participation_rate_cap: float = 0.1
    initial_cash: float = 100_000.0
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1
    reversal_lookback_months: int = 1
    low_vol_lookback_days: int = 126
    universe_lookback_days_dollar_volume: int = 63
    universe_min_price: float = 1.0

    def cost_model(self) -> CostModel:
        return CostModel(
            commission_bps=self.commission_bps,
            half_spread_bps=self.half_spread_bps,
            market_impact_coefficient=self.market_impact_coefficient,
            participation_rate_cap=self.participation_rate_cap,
        )

    def score_fn(self):
        if self.factor == "momentum":
            return lambda accessor, tickers: momentum_score(
                accessor, tickers, self.momentum_lookback_months, self.momentum_skip_months
            )
        if self.factor == "reversal":
            return lambda accessor, tickers: reversal_score(
                accessor, tickers, self.reversal_lookback_months
            )
        return lambda accessor, tickers: low_vol_score(
            accessor, tickers, self.low_vol_lookback_days
        )

    def weights_fn(self):
        if self.construction == "decile_long_short":
            return lambda scores: decile_long_short_weights(scores, n_deciles=self.n_deciles)
        return lambda scores: long_only_top_decile_weights(scores, n_deciles=self.n_deciles)


@dataclass(frozen=True)
class StrategyResult:
    spec: StrategySpec
    nav_history: pd.Series  # indexed by date
    returns: np.ndarray
    sharpe: float
    turnover_total: float
    cost_paid_total: float
    n_rebalances: int
    n_tickers: int
    excluded_tickers: list[str]


def _build_universe_by_date(panel, rebalance_dates, spec: StrategySpec):
    n_deciles_floor = max(2, spec.n_deciles)
    top_n = spec.liquidity_top_n or max(n_deciles_floor * 2, len(panel) // 2)
    universe_table = build_survivorship_aware_universe(
        panel,
        rebalance_dates,
        top_n=top_n,
        lookback_days=spec.universe_lookback_days_dollar_volume,
        min_price=spec.universe_min_price,
    )
    return universe_table.groupby("as_of_date")["ticker"].apply(list).to_dict()


def _build_schedule(panel, universe_by_date, spec: StrategySpec):
    score_fn = spec.score_fn()
    weights_fn = spec.weights_fn()
    schedule = {}
    for as_of, tickers in universe_by_date.items():
        accessor = ConcreteAsOfAccessor(as_of_date=as_of.date(), panel=panel)
        scores = score_fn(accessor, tickers)
        weights = weights_fn(scores)
        if weights:
            schedule[as_of] = weights
    return schedule


def run_strategy(
    spec: StrategySpec,
    panel: dict[str, pd.DataFrame],
    candidates_csv: Path | None = None,
) -> StrategyResult:
    """Run `spec` against `panel` (ticker -> unadjusted OHLCV DataFrame).

    This is the one and only code path that turns a StrategySpec into
    results — both the CLI runner and the dashboard call this function
    directly, so there is no second implementation that could drift.
    """
    excluded: list[str] = []
    if spec.apply_data_quality_filter:
        bad_ticks = set(detect_bad_ticks(panel))
        coverage_gaps = set(detect_coverage_gaps(panel))
        excluded = sorted(bad_ticks | coverage_gaps)
        panel = {t: df for t, df in panel.items() if t not in excluded}

    all_dates = sorted({ts for df in panel.values() for ts in df.index})
    if not all_dates:
        raise ValueError("panel is empty after filtering")

    start, end = all_dates[0].date().isoformat(), all_dates[-1].date().isoformat()
    rebalance_dates = monthly_rebalance_dates(start, end)
    rebalance_dates = [d for d in rebalance_dates if all_dates[0] <= d <= all_dates[-1]]

    universe_by_date = _build_universe_by_date(panel, rebalance_dates, spec)
    schedule = _build_schedule(panel, universe_by_date, spec)

    engine = BacktestEngine(panel, all_dates, schedule, spec.cost_model(), spec.initial_cash)
    history = engine.run()

    navs = pd.Series([r.nav for r in history], index=[r.as_of_date for r in history], name="nav")
    returns = navs.pct_change().dropna().to_numpy()

    return StrategyResult(
        spec=spec,
        nav_history=navs,
        returns=returns,
        sharpe=sharpe_ratio(returns),
        turnover_total=engine.portfolio.total_traded_value,
        cost_paid_total=engine.portfolio.total_cost_paid,
        n_rebalances=len(schedule),
        n_tickers=len(panel),
        excluded_tickers=excluded,
    )


def run_with_survivorship_comparison(
    spec: StrategySpec,
    panel: dict[str, pd.DataFrame],
    candidates_csv: Path,
    as_of: date | None = None,
):
    """Convenience wrapper: also runs the same spec restricted to
    survivors-only and returns the survivorship-sensitivity delta alongside
    the primary result."""
    primary = run_strategy(spec, panel, candidates_csv)

    candidates = load_candidate_universe(candidates_csv, as_of or date.today())
    delisted_set = set(candidates.delisted["ticker"])
    survivors_panel = {t: df for t, df in panel.items() if t not in delisted_set}
    survivors_result = run_strategy(spec, survivors_panel, candidates_csv)

    delta = survivorship_sensitivity(primary.returns, survivors_result.returns)
    return primary, survivors_result, delta
