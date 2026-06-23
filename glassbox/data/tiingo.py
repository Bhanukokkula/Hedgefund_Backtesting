"""TiingoProvider: the price/volume data source.

Tiingo is point-in-time-safe for OHLCV: a price as-of a date is unambiguous.
A single `/prices` call returns both raw (unadjusted) and split/dividend
-adjusted fields plus the split factor and dividend cash for that day, which
is exactly what `glassbox.engine.adjustments` needs to reconstruct an
as-of-correct adjusted series (M2) without ever trusting Tiingo's
fully-adjusted series directly inside simulation code.

Tiingo's fundamentals endpoint (marketCap, shares outstanding) is restricted
to the Dow 30 on the free/Power plan — confirmed by probing a non-Dow ticker
and getting a 400. That is why `get_shares_outstanding` is not implemented
here: this project has no free, full-universe, point-in-time shares
outstanding source. See glassbox.data.fmp.FMPProvider.get_shares_outstanding
and glassbox.factors.fundamental for how that gap is surfaced as a refusal
rather than silently faked.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"


class TiingoRateLimitError(RuntimeError):
    def __init__(self, url: str):
        super().__init__(f"Tiingo hourly request allocation exhausted: {url}")


class TiingoProvider:
    def __init__(self, api_key: str, max_retries: int = 3, retry_backoff_seconds: float = 2.0):
        if not api_key:
            raise ValueError("TiingoProvider requires a non-empty api_key")
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._session = requests.Session()

    def is_point_in_time(self) -> bool:
        return True

    def get_price_history(
        self,
        ticker: str,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Daily OHLCV for `ticker` over [start, end].

        `adjusted` is accepted for protocol compatibility but both raw and
        adjusted columns are always returned (open, high, low, close,
        volume, adj_open, adj_high, adj_low, adj_close, adj_volume,
        div_cash, split_factor) since Tiingo's single endpoint provides
        both in one call — callers needing as-of-correct adjustment should
        reconstruct via glassbox.engine.adjustments, not by trusting
        adj_close directly for dates before today.
        """
        url = f"{TIINGO_BASE_URL}/{ticker}/prices"
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "token": self._api_key,
            "format": "json",
        }
        data = self._get_with_retry(url, params)
        if not data:
            return pd.DataFrame(
                columns=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adj_open",
                    "adj_high",
                    "adj_low",
                    "adj_close",
                    "adj_volume",
                    "div_cash",
                    "split_factor",
                ]
            )
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.rename(
            columns={
                "adjOpen": "adj_open",
                "adjHigh": "adj_high",
                "adjLow": "adj_low",
                "adjClose": "adj_close",
                "adjVolume": "adj_volume",
                "divCash": "div_cash",
                "splitFactor": "split_factor",
            }
        )
        cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "adj_volume",
            "div_cash",
            "split_factor",
        ]
        return df[cols].sort_values("date").reset_index(drop=True)

    def get_universe_symbols(self, as_of: date) -> pd.DataFrame:
        raise NotImplementedError(
            "TiingoProvider has no bulk symbol-list API call; the active/delisted "
            "candidate universe is built once from the cached supported_tickers.csv "
            "via glassbox.data.candidates.load_candidate_universe()."
        )

    def get_delisted_symbols(self) -> pd.DataFrame:
        raise NotImplementedError("see glassbox.data.candidates.load_candidate_universe()")

    def get_shares_outstanding(self, ticker: str, as_of: date) -> float | None:
        raise NotImplementedError(
            "Tiingo fundamentals (marketCap/shares) is Dow-30-only on the free "
            "plan; not usable for a full universe. See FMPProvider and the "
            "fundamental PIT-refusal seam (glassbox.factors.fundamental)."
        )

    def _get_with_retry(self, url: str, params: dict) -> list[dict]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 404:
                    logger.warning("Tiingo 404 for %s (ticker likely delisted/unsupported)", url)
                    return []
                if resp.status_code == 429:
                    # Tiingo's free tier enforces an hourly request allocation, not a
                    # short burst window — retrying within the same call just burns
                    # more of an already-exhausted quota. Surface immediately so the
                    # caller (glassbox.data.ingest) can skip and resume on a later run.
                    logger.warning("Tiingo hourly quota exhausted (429) on %s", url)
                    raise TiingoRateLimitError(url)
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self._retry_backoff_seconds * (attempt + 1))
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Tiingo request failed after {self._max_retries} retries: {url}")
