"""Survivorship-aware monthly universe construction from cached price parquet.

At each monthly rebalance date, the investable set = top-N names by trailing
dollar volume *as known at that date* — including names that later
delisted. No hindsight: a ticker's eventual fate is never consulted when
deciding membership on a given as_of_date; only data dated on or before
as_of_date is read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal


def load_price_panel(prices_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    panel = {}
    for ticker in tickers:
        path = prices_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.set_index("date").sort_index()
        panel[ticker] = df
    return panel


def monthly_rebalance_dates(
    start: str, end: str, calendar_name: str = "NYSE"
) -> list[pd.Timestamp]:
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=start, end_date=end)
    trading_days = pd.DatetimeIndex(schedule.index)
    months = trading_days.to_series().groupby(trading_days.to_period("M")).max()
    return list(pd.DatetimeIndex(months.values))


def build_survivorship_aware_universe(
    panel: dict[str, pd.DataFrame],
    rebalance_dates: list[pd.Timestamp],
    top_n: int,
    lookback_days: int,
    min_price: float,
    max_staleness_days: int = 10,
) -> pd.DataFrame:
    """For each rebalance date, rank tickers by trailing dollar volume using
    only data dated <= as_of_date, and keep the top_n.

    A ticker whose most recent price bar is more than `max_staleness_days`
    before as_of_date is excluded from that snapshot: `df.loc[:as_of]` on a
    ticker with no rows near as_of_date silently returns its last available
    (stale) rows rather than nothing, which would otherwise keep a delisted
    name in the universe indefinitely using its frozen last-known price —
    a real bug caught by tests/test_universe.py's no-hindsight test.

    Returns a long DataFrame indexed by (as_of_date, ticker) with columns:
    close, dollar_volume_avg, rank.
    """
    rows = []
    for as_of in rebalance_dates:
        candidates = []
        for ticker, df in panel.items():
            window = df.loc[:as_of].tail(lookback_days)
            if window.empty:
                continue
            if (as_of - window.index[-1]).days > max_staleness_days:
                continue
            last_close = window["adj_close"].iloc[-1]
            last_raw_close = window["close"].iloc[-1]
            if last_raw_close < min_price:
                continue
            dollar_volume = (window["close"] * window["volume"]).mean()
            if pd.isna(dollar_volume) or dollar_volume <= 0:
                continue
            candidates.append((ticker, last_close, dollar_volume))

        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda x: x[2], reverse=True)[:top_n]
        for rank, (ticker, close, dollar_volume) in enumerate(ranked, start=1):
            rows.append(
                {
                    "as_of_date": as_of,
                    "ticker": ticker,
                    "close": close,
                    "dollar_volume_avg": dollar_volume,
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def write_universe_snapshots(universe: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(out_path, index=False)
