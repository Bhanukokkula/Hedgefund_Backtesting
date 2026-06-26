"""M1 stop-and-report gate: does the free data actually support the
survivorship thesis?

Produces:
  - delisted vs surviving ticker counts in the sampled universe
  - % of universe-months that include eventually-delisted names
  - coverage gaps / suspicious NaN runs / obvious bad ticks
  - survivorship delta: survivors-only vs full-universe mean return & vol

Run as: .venv/bin/python -m glassbox.validation.m1_report
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from glassbox.data.universe import (
    build_survivorship_aware_universe,
    load_price_panel,
    monthly_rebalance_dates,
)
from glassbox.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class M1ValidationReport:
    n_active_sampled: int
    n_delisted_sampled: int
    n_rebalance_dates: int
    n_universe_rows: int
    pct_universe_months_with_delisted: float
    n_universe_months_with_delisted: int
    coverage_gap_tickers: list[str]
    bad_tick_tickers: list[str]
    full_universe_mean_daily_return: float
    full_universe_daily_vol: float
    survivors_only_mean_daily_return: float
    survivors_only_daily_vol: float
    survivorship_return_delta_bps_annualized: float
    survivorship_vol_delta_bps_annualized: float
    gate_threshold_pct: float
    gate_passed: bool


def detect_coverage_gaps(panel: dict[str, pd.DataFrame], max_gap_days: int = 10) -> list[str]:
    flagged = []
    for ticker, df in panel.items():
        if df.empty:
            continue
        gaps = df.index.to_series().diff().dt.days
        if (gaps > max_gap_days).any():
            flagged.append(ticker)
    return flagged


def detect_bad_ticks(panel: dict[str, pd.DataFrame], max_daily_move: float = 0.95) -> list[str]:
    flagged = []
    for ticker, df in panel.items():
        if len(df) < 2:
            continue
        ret = df["adj_close"].pct_change().abs()
        if (ret > max_daily_move).any():
            flagged.append(ticker)
    return flagged


def detect_stale_pricing(
    panel: dict[str, pd.DataFrame], max_zero_return_fraction: float = 0.25
) -> list[str]:
    """Flag tickers whose price barely moves day to day — a sign of thin or
    stale trading, not genuine stability. A name with this many days of
    exactly-zero return is artificially low-volatility, which a naive
    low-vol factor would systematically over-select; this is the second
    real contamination source found auditing that factor's real-data
    result, after preferred shares (see glassbox.data.candidates).
    """
    flagged = []
    for ticker, df in panel.items():
        if len(df) < 2:
            continue
        returns = df["adj_close"].pct_change().dropna()
        if returns.empty:
            continue
        zero_fraction = (returns == 0).mean()
        if zero_fraction > max_zero_return_fraction:
            flagged.append(ticker)
    return flagged


def _annualized_return(daily_mean: float) -> float:
    return daily_mean * 252 * 10_000  # bps


def _annualized_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(252) * 10_000  # bps


def run_m1_validation() -> M1ValidationReport:
    sample = pd.read_parquet(settings.parquet_dir / "candidate_sample.parquet")
    prices_dir = settings.parquet_dir / "prices"
    available_tickers = [p.stem for p in prices_dir.glob("*.parquet")]
    sample = sample[sample["ticker"].isin(available_tickers)].reset_index(drop=True)

    n_active = int((~sample["is_delisted"]).sum())
    n_delisted = int(sample["is_delisted"].sum())
    delisted_tickers = set(sample.loc[sample["is_delisted"], "ticker"])

    panel = load_price_panel(prices_dir, sample["ticker"].tolist())

    all_dates = pd.concat([df.index.to_series() for df in panel.values()])
    start = all_dates.min().date().isoformat()
    end = all_dates.max().date().isoformat()
    rebalance_dates = monthly_rebalance_dates(start, end, settings.calendar.name)

    universe = build_survivorship_aware_universe(
        panel,
        rebalance_dates,
        top_n=settings.universe.top_n_by_dollar_volume,
        lookback_days=settings.universe.lookback_days_dollar_volume,
        min_price=settings.universe.min_price,
    )

    if universe.empty:
        raise RuntimeError("survivorship-aware universe construction produced zero rows")

    universe["is_delisted"] = universe["ticker"].isin(delisted_tickers)
    months_with_delisted = universe.groupby("as_of_date")["is_delisted"].any()
    n_months_with_delisted = int(months_with_delisted.sum())
    pct_months_with_delisted = float(months_with_delisted.mean())

    coverage_gaps = detect_coverage_gaps(panel)
    bad_ticks = detect_bad_ticks(panel)

    full_returns = _universe_daily_returns(panel, universe)
    survivors_universe = universe[~universe["is_delisted"]]
    survivors_returns = _universe_daily_returns(panel, survivors_universe)

    full_mean, full_vol = full_returns.mean(), full_returns.std()
    surv_mean, surv_vol = survivors_returns.mean(), survivors_returns.std()

    gate_threshold = settings.validation.min_pct_universe_months_with_delisted
    report = M1ValidationReport(
        n_active_sampled=n_active,
        n_delisted_sampled=n_delisted,
        n_rebalance_dates=len(rebalance_dates),
        n_universe_rows=len(universe),
        pct_universe_months_with_delisted=pct_months_with_delisted,
        n_universe_months_with_delisted=n_months_with_delisted,
        coverage_gap_tickers=sorted(coverage_gaps),
        bad_tick_tickers=sorted(bad_ticks),
        full_universe_mean_daily_return=float(full_mean),
        full_universe_daily_vol=float(full_vol),
        survivors_only_mean_daily_return=float(surv_mean),
        survivors_only_daily_vol=float(surv_vol),
        survivorship_return_delta_bps_annualized=float(
            _annualized_return(surv_mean) - _annualized_return(full_mean)
        ),
        survivorship_vol_delta_bps_annualized=float(
            _annualized_vol(surv_vol) - _annualized_vol(full_vol)
        ),
        gate_threshold_pct=gate_threshold,
        gate_passed=pct_months_with_delisted >= gate_threshold,
    )
    return report


def _universe_daily_returns(panel: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.Series:
    """Equal-weight daily returns of the monthly-rebalanced membership in `universe`."""
    membership_by_month = universe.groupby("as_of_date")["ticker"].apply(set)
    sorted_dates = sorted(membership_by_month.index)
    all_returns = []
    for i, as_of in enumerate(sorted_dates):
        members = membership_by_month[as_of]
        period_end = sorted_dates[i + 1] if i + 1 < len(sorted_dates) else None
        for ticker in members:
            df = panel.get(ticker)
            if df is None:
                continue
            window = df.loc[as_of:period_end] if period_end is not None else df.loc[as_of:]
            ret = window["adj_close"].pct_change().dropna()
            all_returns.append(ret)
    if not all_returns:
        return pd.Series(dtype=float)
    combined = pd.concat(all_returns)
    return combined.groupby(combined.index).mean()


def save_report(report: M1ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
        f.write("\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_m1_validation()
    out_path = settings.cache_dir / "m1_validation_report.json"
    save_report(report, out_path)
    print(json.dumps(asdict(report), indent=2, default=str))
    print(f"\nWritten to {out_path}")
    if not report.gate_passed:
        print("\n*** STOP-AND-REPORT: delisted coverage below gate threshold ***")


if __name__ == "__main__":
    main()
