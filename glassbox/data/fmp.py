"""FMPProvider: supplementary delisting metadata and (flagged) fundamentals.

As of this build, FMP's free tier no longer offers the bulk active/delisted
symbol-list endpoints this project's spec assumed (`stock/list`,
`sp500-constituent`, etc. return empty or "Restricted Endpoint"). Two
endpoints do work on the free key and are used here:

  - `stable/delisted-companies` (page=0 only, ~100 most recent records) —
    used as a best-effort cross-check/enrichment for delisting reasons on
    the subset of names it covers. It is NOT the primary source for the
    survivorship-aware universe; that's glassbox.data.candidates, built
    from Tiingo's public supported_tickers.csv.
  - `stable/shares-float` — gives a CURRENT snapshot of shares outstanding
    only, not a historical series. Using it for a past date would itself be
    a look-ahead violation (today's share count applied to 2010's market
    cap). That is exactly the gap glassbox.factors.fundamental's PIT-refusal
    seam exists for: `is_point_in_time()` returns False here, and any
    caller building the Size factor on this data must go through that
    refusal path rather than trusting it silently.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"


class FMPProvider:
    def __init__(self, api_key: str, max_retries: int = 3, retry_backoff_seconds: float = 2.0):
        if not api_key:
            raise ValueError("FMPProvider requires a non-empty api_key")
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._session = requests.Session()

    def is_point_in_time(self) -> bool:
        return False

    def get_price_history(
        self, ticker: str, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame:
        raise NotImplementedError("FMPProvider is not used for price history; see TiingoProvider.")

    def get_universe_symbols(self, as_of: date) -> pd.DataFrame:
        raise NotImplementedError(
            "FMP's bulk active-symbol-list endpoints are paywalled on the free tier "
            "as of this build; see glassbox.data.candidates for the Tiingo-based substitute."
        )

    def get_delisted_symbols(self, limit: int = 100) -> pd.DataFrame:
        """Most recent delisted companies (free tier: page=0 only, ~100 rows).

        Best-effort enrichment only — not load-bearing for the M1 survivorship
        gate, which uses glassbox.data.candidates instead.
        """
        url = f"{FMP_BASE_URL}/delisted-companies"
        params = {"limit": limit, "page": 0, "apikey": self._api_key}
        data = self._get_with_retry(url, params)
        if not data:
            return pd.DataFrame(
                columns=["symbol", "companyName", "exchange", "ipoDate", "delistedDate"]
            )
        return pd.DataFrame(data)

    def get_shares_outstanding(self, ticker: str, as_of: date) -> float | None:
        """Current shares-outstanding snapshot. NOT point-in-time for any
        as_of in the past — callers must treat this as a non-PIT input.
        """
        url = f"{FMP_BASE_URL}/shares-float"
        params = {"symbol": ticker, "apikey": self._api_key}
        data = self._get_with_retry(url, params)
        if not data:
            return None
        return data[0].get("outstandingShares")

    def _get_with_retry(self, url: str, params: dict) -> list[dict]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict) and (
                        "Error Message" in payload or "Premium Query Parameter" in str(payload)
                    ):
                        logger.warning("FMP restricted/error response: %s", payload)
                        return []
                    return payload
                if resp.status_code == 429:
                    wait = self._retry_backoff_seconds * (attempt + 1)
                    logger.warning("FMP rate-limited, backing off %.1fs", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self._retry_backoff_seconds * (attempt + 1))
        if last_exc:
            raise last_exc
        raise RuntimeError(f"FMP request failed after {self._max_retries} retries: {url}")
