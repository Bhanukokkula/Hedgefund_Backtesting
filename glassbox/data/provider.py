"""The DataProvider protocol: the only seam through which network data enters
GLASSBOX.

Every external data source (Tiingo, FMP, ...) implements this protocol.
Everything downstream of ingestion — the AsOfAccessor, the engine, the
factor library — depends only on this protocol (or on LocalParquetProvider,
which serves cached data with the same shape). No module outside
`glassbox.data` may import an HTTP client directly: `rg "import requests"
glassbox/` should only ever match files in `glassbox/data/`.

Concrete implementations are built in M1. This module fixes the contract
first so every later milestone can depend on a stable shape.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataProvider(Protocol):
    """Source of point-in-time-able market data.

    Implementations must NOT silently restate history: a value returned for
    `as_of=d` must be the value as it was knowable on date `d`, not a value
    later revised. Where a source cannot make that guarantee (e.g. free
    fundamentals data that gets restated), the provider must surface that
    via `is_point_in_time() -> False` so callers (see
    glassbox.factors.fundamental) can flag/refuse the input instead of
    silently trusting it.
    """

    def is_point_in_time(self) -> bool:
        """Whether this provider's data is restatement-aware point-in-time.

        Price/volume/shares-outstanding from Tiingo/FMP: True (a price as-of
        a date is unambiguous). Free fundamentals (most income-statement /
        balance-sheet feeds): False — they reflect the latest-known restated
        figure, not what was reported as-of the original filing date.
        """
        ...

    def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        adjusted: bool,
    ) -> pd.DataFrame:
        """Daily OHLCV for `ticker` over [start, end].

        Must return columns: open, high, low, close, volume, date (index or
        column). `adjusted=False` returns raw split/dividend-unadjusted
        prices; `adjusted=True` returns prices adjusted for splits and
        dividends using ALL adjustment events regardless of date — callers
        needing an as-of-correct adjusted series must reconstruct it via
        AsOfAccessor (glassbox.engine.asof), never by calling this with
        adjusted=True and trusting it directly inside simulation code.
        """
        ...

    def get_universe_symbols(self, as_of: date) -> pd.DataFrame:
        """All US equity symbols that were listed and trading as of `as_of`.

        Must include names that listed before `as_of` and have not yet
        delisted as of `as_of` — and must NOT use hindsight (i.e. must not
        silently exclude names that will delist later). Returns at least
        columns: ticker, ipo_date, delisting_date (NaT if still active as of
        the data pull).
        """
        ...

    def get_delisted_symbols(self) -> pd.DataFrame:
        """All historically delisted US equity symbols with delisting dates.

        Returns at least columns: ticker, ipo_date, delisting_date,
        delisting_reason (where available).
        """
        ...

    def get_shares_outstanding(self, ticker: str, as_of: date) -> float | None:
        """Shares outstanding for `ticker` as knowable on `as_of`.

        Used for market-cap (size factor). Subject to a publication lag
        (see config.yaml: publication_lag.shares_outstanding_days) applied
        by the AsOfAccessor, not by the provider itself — the provider
        returns the raw reported value and its report date; lag enforcement
        is the accessor's job so it stays in one place.
        """
        ...


class LocalParquetProvider:
    """Serves cached parquet snapshots with the same shape as DataProvider.

    Implemented in M1. After the first ingestion run, this is the only
    provider the rest of the codebase (engine, factors, validation) talks
    to — the network is touched exclusively by TiingoProvider/FMPProvider
    inside `glassbox/data/ingest.py`.
    """
