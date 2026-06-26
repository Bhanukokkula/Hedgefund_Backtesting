"""Provider tests using a mocked HTTP session — no network access, per the
project rule that everything outside the ingestion script itself must be
testable offline."""

from __future__ import annotations

from datetime import date

import pytest

from glassbox.data.fmp import FMPProvider
from glassbox.data.tiingo import TiingoMonthlyQuotaError, TiingoProvider, TiingoRateLimitError


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if not isinstance(payload, str) else payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.response


def test_tiingo_is_point_in_time():
    provider = TiingoProvider(api_key="x")
    assert provider.is_point_in_time() is True


def test_fmp_is_not_point_in_time():
    provider = FMPProvider(api_key="x")
    assert provider.is_point_in_time() is False


def test_tiingo_get_price_history_renames_and_orders_columns():
    payload = [
        {
            "date": "2020-01-02T00:00:00.000Z",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100,
            "adjOpen": 1.0,
            "adjHigh": 2.0,
            "adjLow": 0.5,
            "adjClose": 1.5,
            "adjVolume": 100,
            "divCash": 0.0,
            "splitFactor": 1.0,
        }
    ]
    provider = TiingoProvider(api_key="x")
    provider._session = _FakeSession(_FakeResponse(200, payload))
    df = provider.get_price_history("AAA", date(2020, 1, 1), date(2020, 1, 5), adjusted=True)
    assert list(df.columns) == [
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
    assert len(df) == 1


def test_tiingo_404_returns_empty_dataframe():
    provider = TiingoProvider(api_key="x")
    provider._session = _FakeSession(_FakeResponse(404, {"detail": "Not found"}))
    df = provider.get_price_history("ZZZ", date(2020, 1, 1), date(2020, 1, 5), adjusted=True)
    assert df.empty


def test_tiingo_monthly_quota_message_raises_distinct_error_not_treated_as_bad_ticker():
    """Regression test: Tiingo's monthly unique-symbol cap comes back as
    HTTP 200 with a plain-text body, not a 4xx. An earlier version of
    glassbox.data.ingest let this fall through to a generic JSONDecodeError
    and permanently blacklisted the ticker as if it had no data — corrupting
    the skip-list with perfectly good, simply-not-yet-looked-up tickers.
    This must raise a distinct, identifiable error instead."""
    provider = TiingoProvider(api_key="x")
    quota_message = (
        "You have run over your 500 symbol look up for this month. "
        "Please upgrade at https://api.tiingo.com/pricing to have your limits increased."
    )
    fake_session = _FakeSession(_FakeResponse(200, quota_message))
    provider._session = fake_session
    with pytest.raises(TiingoMonthlyQuotaError):
        provider.get_price_history("ZBRA", date(2020, 1, 1), date(2020, 1, 5), adjusted=True)
    # Must not retry — same as the hourly rate limit, retrying can't help.
    assert len(fake_session.calls) == 1


def test_tiingo_429_raises_rate_limit_error_without_retrying():
    provider = TiingoProvider(api_key="x")
    fake_session = _FakeSession(_FakeResponse(429, {"detail": "over allocation"}))
    provider._session = fake_session
    with pytest.raises(TiingoRateLimitError):
        provider.get_price_history("AAA", date(2020, 1, 1), date(2020, 1, 5), adjusted=True)
    # Must not hammer the rate-limited endpoint with retries.
    assert len(fake_session.calls) == 1


def test_fmp_delisted_companies_handles_restricted_response():
    provider = FMPProvider(api_key="x")
    provider._session = _FakeSession(_FakeResponse(200, {"Error Message": "Restricted Endpoint"}))
    df = provider.get_delisted_symbols()
    assert df.empty


def test_fmp_shares_outstanding_parses_payload():
    provider = FMPProvider(api_key="x")
    provider._session = _FakeSession(_FakeResponse(200, [{"outstandingShares": 14_687_356_000}]))
    shares = provider.get_shares_outstanding("AAPL", date(2026, 6, 22))
    assert shares == 14_687_356_000


def test_fmp_shares_outstanding_returns_none_when_empty():
    provider = FMPProvider(api_key="x")
    provider._session = _FakeSession(_FakeResponse(200, []))
    assert provider.get_shares_outstanding("ZZZ", date(2026, 6, 22)) is None
