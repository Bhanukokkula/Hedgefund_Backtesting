"""The AsOfAccessor: the single chokepoint enforcing the core invariant.

    No computation may read any datum whose knowable-date is later than the
    simulation's current as-of clock.

Every read of price, volume, shares-outstanding, or factor input by
strategy code, the portfolio constructor, or the cost model goes through an
AsOfAccessor bound to one as_of_date. There is no method anywhere in this
codebase that returns a full future-inclusive series to strategy code — if
a strategy wants "the price series", it gets the series truncated at
as_of_date, full stop.

Implementation lands in M2. This module fixes the contract first (M0) so
M1's ingestion code and M3's engine can both be written against a frozen
interface. The adversarial test suite (tests/test_asof_adversarial.py,
written in M2) is what makes this guarantee real: tests that deliberately
try to read tomorrow's close, or a same-day fully-adjusted price, and
assert the accessor refuses.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class AsOfAccessor(Protocol):
    """Bound to one as_of_date; every method is implicitly clamped to it."""

    as_of_date: date

    def price_series(
        self,
        ticker: str,
        lookback_days: int,
        field: str = "close",
        adjusted: bool = True,
    ) -> pd.Series:
        """Last `lookback_days` of `field` for `ticker`, ending at as_of_date.

        Raises if `lookback_days` would require data before the ticker's
        IPO date (no synthetic backfill) — callers must handle short
        histories explicitly rather than receiving silently-padded NaNs
        that could be mistaken for real (lack of) movement.

        If `adjusted=True`, the returned series reflects only split/dividend
        events with an ex-date <= as_of_date. It is NEVER the fully-adjusted
        series Tiingo would return today for a query in the past — that
        would leak knowledge of future corporate actions into a historical
        as-of view. See glassbox.engine.adjustments for how the as-of
        adjustment factor is built from unadjusted prices.
        """
        ...

    def latest_price(self, ticker: str, field: str = "close") -> float:
        """Single most recent knowable `field` value for `ticker` as of as_of_date."""
        ...

    def shares_outstanding(self, ticker: str) -> float | None:
        """Shares outstanding knowable as of as_of_date.

        Enforces config.yaml: publication_lag.shares_outstanding_days —
        a figure reported on day D is not knowable until D + lag.
        """
        ...

    def market_cap(self, ticker: str) -> float | None:
        """latest_price(ticker) * shares_outstanding(ticker), or None if either is missing."""
        ...

    def universe(self) -> list[str]:
        """Tickers investable as of as_of_date under the survivorship-aware universe
        definition (top-N by trailing dollar volume, including names that will
        later delist — see glassbox.data.universe, built in M1)."""
        ...

    def is_tradable(self, ticker: str) -> bool:
        """Whether `ticker` has a live, non-halted quote as of as_of_date
        (i.e. not yet delisted, not a corporate-action-suspended session)."""
        ...

    def advance_to(self, new_as_of_date: date) -> AsOfAccessor:
        """Return a new accessor bound to `new_as_of_date`.

        `new_as_of_date` must be >= self.as_of_date — the clock is
        monotonic; there is no method to move it backwards. This is the
        only sanctioned way to change the bound date; AsOfAccessor instances
        are otherwise immutable.
        """
        ...
