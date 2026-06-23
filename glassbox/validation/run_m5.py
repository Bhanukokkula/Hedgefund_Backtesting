"""Run the real M5 lie-resistance suite against cached price data.

Uses every ticker cached under data/parquet/prices (not just the 120-ticker
target sample — earlier ingestion runs with different random seeds left
extra real tickers cached, and there's no reason to throw away real data),
classified active/delisted via glassbox.data.candidates against the same
Tiingo supported_tickers.csv used for the M1 gate.

This universe is small relative to the project's eventual target
(config.yaml: top_n_by_dollar_volume=500) — it's bounded by Tiingo's
free-tier hourly request allocation, not by design. Results here are real,
not synthetic, but should be read as a demonstration on a constrained
sample, not a production-scale backtest. That limitation is reported
explicitly, not hidden.

Run as: .venv/bin/python -m glassbox.validation.run_m5
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from glassbox.data.candidates import load_candidate_universe
from glassbox.data.universe import (
    build_survivorship_aware_universe,
    load_price_panel,
    monthly_rebalance_dates,
)
from glassbox.engine.asof_accessor import ConcreteAsOfAccessor
from glassbox.engine.backtest import BacktestEngine
from glassbox.engine.costs import CostModel
from glassbox.factors.ranking import long_only_top_decile_weights
from glassbox.factors.scoring import low_vol_score, momentum_score, reversal_score
from glassbox.settings import settings
from glassbox.validation.m1_report import detect_bad_ticks, detect_coverage_gaps
from glassbox.validation.metrics import (
    deflated_sharpe_ratio,
    harvey_liu_haircut_sharpe,
    sharpe_ratio,
)
from glassbox.validation.sensitivity import cost_sensitivity, survivorship_sensitivity
from glassbox.validation.walk_forward import run_walk_forward

logger = logging.getLogger(__name__)

NET_COST_MODEL = CostModel(
    commission_bps=settings.costs.commission_bps,
    half_spread_bps=settings.costs.half_spread_bps,
    market_impact_coefficient=settings.costs.market_impact_coefficient,
    participation_rate_cap=settings.costs.participation_rate_cap,
)
ZERO_COST_MODEL = CostModel(0.0, 0.0, 0.0, 1.0)

FACTOR_SCORE_FNS = {
    "momentum": lambda accessor, tickers: momentum_score(
        accessor,
        tickers,
        settings.factors.momentum.lookback_months,
        settings.factors.momentum.skip_months,
    ),
    "reversal": lambda accessor, tickers: reversal_score(
        accessor, tickers, settings.factors.reversal.lookback_months
    ),
    "low_vol": lambda accessor, tickers: low_vol_score(
        accessor, tickers, settings.factors.low_vol.lookback_days
    ),
}
N_TRIALS = len(FACTOR_SCORE_FNS)


@dataclass
class FactorResult:
    factor: str
    n_tickers: int
    n_rebalances: int
    gross_sharpe: float
    net_sharpe: float
    cost_drag_bps: float
    full_universe_sharpe: float
    survivors_only_sharpe: float
    survivorship_sharpe_inflation: float
    survivorship_return_inflation_bps: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    naive_sharpe: float
    deflated_sharpe: float
    haircut_sharpe: float


def _build_schedule(panel, universe_by_date, score_fn, n_deciles):
    """Long-only top-decile, not decile long-short. The dollar-neutral
    long-short construction (100% long / 100% short, 200% gross) blew this
    sample's NAV negative under monthly full-decile rotation — this engine
    has no margin/leverage model, so an unconstrained short leg compounding
    against a noisy, sparse small/micro-cap sample is a real, reproducible
    failure mode, not a fluke (confirmed: NAV went negative on 2015-06-26
    and decayed toward zero by the end of the sample). Long-only avoids it
    structurally (fully invested, no shorting, NAV cannot go negative) and
    is one of the two constructions M4 was explicitly built to support.

    `universe_by_date` restricts each rebalance to that date's
    dollar-volume-ranked investable set (see build_survivorship_aware_universe)
    rather than every cached ticker indiscriminately — without this, a single
    illiquid micro-cap's idiosyncratic bounce can dominate a decile and
    produce an implausible NAV path even with no software bug involved
    (confirmed: reversal's gross NAV hit 2,820x on the unfiltered universe).
    """
    schedule = {}
    for as_of, tickers in universe_by_date.items():
        accessor = ConcreteAsOfAccessor(as_of_date=as_of.date(), panel=panel)
        scores = score_fn(accessor, tickers)
        weights = long_only_top_decile_weights(scores, n_deciles=n_deciles)
        if weights:
            schedule[as_of] = weights
    return schedule


def _returns_from_schedule(panel, trading_dates, schedule, cost_model, initial_cash=100_000.0):
    engine = BacktestEngine(panel, trading_dates, schedule, cost_model, initial_cash)
    history = engine.run()
    navs = pd.Series([r.nav for r in history])
    return navs.pct_change().dropna().to_numpy()


def run_all_factors() -> list[FactorResult]:
    prices_dir = settings.parquet_dir / "prices"
    all_tickers = [p.stem for p in prices_dir.glob("*.parquet")]
    panel = load_price_panel(prices_dir, all_tickers)
    panel = {t: df for t, df in panel.items() if len(df) >= 252}  # at least 1yr of history

    # M1's validation report already flagged tickers with implausible
    # one-day moves (>95%) and large coverage gaps. Feeding those straight
    # into momentum/reversal — which actively seek out extreme movers — is
    # exactly how a single bad tick turns into a NAV that compounds to
    # negative quadrillions: confirmed by an earlier run of this script
    # before this filter existed. Excluding them here is the same
    # data-quality gate M1 already proved necessary, applied consistently.
    bad_ticks = set(detect_bad_ticks(panel))
    coverage_gaps = set(detect_coverage_gaps(panel))
    excluded = bad_ticks | coverage_gaps
    if excluded:
        logger.warning(
            "excluding %d flagged tickers from factor universe: %s", len(excluded), sorted(excluded)
        )
    panel = {t: df for t, df in panel.items() if t not in excluded}

    candidates = load_candidate_universe(
        settings.raw_dir / "tiingo_supported_tickers.csv", date.today()
    )
    delisted_set = set(candidates.delisted["ticker"])

    all_dates = sorted({ts for df in panel.values() for ts in df.index})
    start, end = all_dates[0].date().isoformat(), all_dates[-1].date().isoformat()
    rebalance_dates = monthly_rebalance_dates(start, end, settings.calendar.name)
    rebalance_dates = [d for d in rebalance_dates if d >= all_dates[0] and d <= all_dates[-1]]

    n_deciles = max(2, min(10, len(panel) // 10))

    # Restrict each month to the top half of the panel by trailing dollar
    # volume — the liquidity filter M1's universe construction already
    # implements. Without it, a single thinly-traded micro-cap can dominate
    # a decile and produce an implausible NAV path with no software bug
    # involved (confirmed: reversal's gross NAV hit 2,820x unfiltered).
    liquidity_top_n = max(n_deciles * 2, len(panel) // 2)
    universe_table = build_survivorship_aware_universe(
        panel,
        rebalance_dates,
        top_n=liquidity_top_n,
        lookback_days=settings.universe.lookback_days_dollar_volume,
        min_price=settings.universe.min_price,
    )
    universe_by_date = universe_table.groupby("as_of_date")["ticker"].apply(list).to_dict()

    survivors_universe_by_date = {
        d: [t for t in month_tickers if t not in delisted_set]
        for d, month_tickers in universe_by_date.items()
    }

    results = []
    for factor_name, score_fn in FACTOR_SCORE_FNS.items():
        logger.info("running factor: %s", factor_name)
        schedule = _build_schedule(panel, universe_by_date, score_fn, n_deciles)
        if not schedule:
            logger.warning("factor %s produced no usable rebalances, skipping", factor_name)
            continue

        net_returns = _returns_from_schedule(panel, all_dates, schedule, NET_COST_MODEL)
        gross_returns = _returns_from_schedule(panel, all_dates, schedule, ZERO_COST_MODEL)
        cost_sens = cost_sensitivity(gross_returns, net_returns)

        survivors_schedule = _build_schedule(panel, survivors_universe_by_date, score_fn, n_deciles)
        survivors_returns = (
            _returns_from_schedule(panel, all_dates, survivors_schedule, NET_COST_MODEL)
            if survivors_schedule
            else net_returns
        )
        surv_sens = survivorship_sensitivity(net_returns, survivors_returns)

        wf = run_walk_forward(panel, schedule, 0.7, NET_COST_MODEL, 100_000.0)

        naive_sharpe = sharpe_ratio(net_returns)
        # DSR/haircut require the Sharpe and observation count at the SAME
        # frequency — feeding the annualized Sharpe with a daily n_obs
        # inflates sqrt(n_obs-1) ~100x and pins every factor's DSR near 1.0
        # regardless of quality (see glassbox.validation.metrics docstring).
        daily_sharpe = sharpe_ratio(net_returns, periods_per_year=1)
        dsr = deflated_sharpe_ratio(daily_sharpe, n_trials=N_TRIALS, n_obs=len(net_returns))
        haircut_daily = harvey_liu_haircut_sharpe(
            daily_sharpe, n_trials=N_TRIALS, n_obs=len(net_returns)
        )
        haircut = haircut_daily * np.sqrt(252)  # re-annualize for display next to naive_sharpe

        results.append(
            FactorResult(
                factor=factor_name,
                n_tickers=len(panel),
                n_rebalances=len(schedule),
                gross_sharpe=cost_sens.gross_sharpe,
                net_sharpe=cost_sens.net_sharpe,
                cost_drag_bps=cost_sens.cost_drag_bps,
                full_universe_sharpe=surv_sens.full_universe_sharpe,
                survivors_only_sharpe=surv_sens.survivors_only_sharpe,
                survivorship_sharpe_inflation=surv_sens.sharpe_inflation,
                survivorship_return_inflation_bps=surv_sens.return_inflation_bps,
                in_sample_sharpe=wf.in_sample_sharpe,
                out_of_sample_sharpe=wf.out_of_sample_sharpe,
                naive_sharpe=naive_sharpe,
                deflated_sharpe=dsr,
                haircut_sharpe=haircut,
            )
        )
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_all_factors()
    out_path = settings.cache_dir / "m5_factor_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
        f.write("\n")
    print(json.dumps([asdict(r) for r in results], indent=2, default=str))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
