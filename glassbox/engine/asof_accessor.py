"""Concrete AsOfAccessor: enforces the core invariant in code, not just docs.

    No computation may read any datum whose knowable-date is later than the
    simulation's current as-of clock.

Backed by a panel of UNADJUSTED price DataFrames (one per ticker, columns:
close, open, high, low, volume, split_factor, div_cash) plus a survivorship
-aware universe table (see glassbox.data.universe). Every read truncates to
rows with date <= as_of_date before doing anything else — there is no code
path in this class that can see a row dated after as_of_date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from glassbox.engine.adjustments import reconstruct_asof_adjusted_close


@dataclass(frozen=True)
class ConcreteAsOfAccessor:
    as_of_date: date
    panel: dict[str, pd.DataFrame]
    universe_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    shares_outstanding_map: dict[str, float] | None = None

    def _truncated(self, ticker: str, lookback_days: int | None = None) -> pd.DataFrame:
        df = self.panel.get(ticker)
        if df is None:
            return pd.DataFrame()
        cutoff = pd.Timestamp(self.as_of_date)
        window = df.loc[df.index <= cutoff]
        if lookback_days is not None:
            window = window.tail(lookback_days)
        return window

    def price_series(
        self,
        ticker: str,
        lookback_days: int,
        field: str = "close",
        adjusted: bool = True,
    ) -> pd.Series:
        window = self._truncated(ticker, lookback_days)
        if window.empty:
            return pd.Series(dtype=float)
        if not adjusted:
            if field not in window.columns:
                raise KeyError(f"unknown field '{field}' for {ticker}")
            return window[field]
        if field != "close":
            raise ValueError("adjusted=True is only defined for field='close'")
        return reconstruct_asof_adjusted_close(window)

    def latest_price(self, ticker: str, field: str = "close") -> float | None:
        window = self._truncated(ticker)
        if window.empty:
            return None
        if field == "close":
            return float(reconstruct_asof_adjusted_close(window).iloc[-1])
        return float(window[field].iloc[-1])

    def shares_outstanding(self, ticker: str) -> float | None:
        if self.shares_outstanding_map is None:
            return None
        return self.shares_outstanding_map.get(ticker)

    def market_cap(self, ticker: str) -> float | None:
        price = self.latest_price(ticker)
        shares = self.shares_outstanding(ticker)
        if price is None or shares is None:
            return None
        return price * shares

    def universe(self) -> list[str]:
        if self.universe_table.empty:
            return []
        cutoff = pd.Timestamp(self.as_of_date)
        rows = self.universe_table[self.universe_table["as_of_date"] <= cutoff]
        if rows.empty:
            return []
        latest_as_of = rows["as_of_date"].max()
        return sorted(rows.loc[rows["as_of_date"] == latest_as_of, "ticker"].tolist())

    def is_tradable(self, ticker: str, max_staleness_days: int = 10) -> bool:
        window = self._truncated(ticker, lookback_days=1)
        if window.empty:
            return False
        return (pd.Timestamp(self.as_of_date) - window.index[-1]).days <= max_staleness_days

    def advance_to(self, new_as_of_date: date) -> ConcreteAsOfAccessor:
        if new_as_of_date < self.as_of_date:
            raise ValueError(
                f"clock is monotonic: cannot advance from {self.as_of_date} "
                f"backward to {new_as_of_date}"
            )
        return ConcreteAsOfAccessor(
            as_of_date=new_as_of_date,
            panel=self.panel,
            universe_table=self.universe_table,
            shares_outstanding_map=self.shares_outstanding_map,
        )
