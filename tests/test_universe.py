"""Synthetic-fixture tests for the survivorship-aware universe builder.

These prove the no-hindsight property directly: a name that later delists
must still appear in earlier monthly snapshots taken while it was trading,
and must vanish from snapshots once its price data ends — without any
membership decision depending on knowledge of the eventual delisting.
"""

from __future__ import annotations

import pandas as pd

from glassbox.data.universe import build_survivorship_aware_universe


def _make_price_df(dates, price, volume):
    return pd.DataFrame(
        {
            "close": price,
            "adj_close": price,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates),
    )


def test_delisted_name_appears_before_delisting_and_disappears_after():
    dates_full = pd.date_range("2020-01-01", "2020-06-30", freq="B")
    dates_delisted = pd.date_range("2020-01-01", "2020-03-31", freq="B")

    panel = {
        "SURVIVOR": _make_price_df(dates_full, price=10.0, volume=1_000_000),
        "DELISTED": _make_price_df(dates_delisted, price=10.0, volume=2_000_000),
    }

    rebalance_dates = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-04-30"),
    ]

    universe = build_survivorship_aware_universe(
        panel, rebalance_dates, top_n=2, lookback_days=20, min_price=1.0
    )

    jan_tickers = set(universe.loc[universe.as_of_date == pd.Timestamp("2020-01-31"), "ticker"])
    feb_tickers = set(universe.loc[universe.as_of_date == pd.Timestamp("2020-02-28"), "ticker"])
    apr_tickers = set(universe.loc[universe.as_of_date == pd.Timestamp("2020-04-30"), "ticker"])

    assert "DELISTED" in jan_tickers
    assert "DELISTED" in feb_tickers
    assert "DELISTED" not in apr_tickers  # no price data after 2020-03-31: correctly absent
    assert "SURVIVOR" in apr_tickers


def test_top_n_ranking_by_trailing_dollar_volume():
    dates = pd.date_range("2020-01-01", "2020-01-31", freq="B")
    panel = {
        "HIGH_VOL": _make_price_df(dates, price=10.0, volume=10_000_000),
        "MID_VOL": _make_price_df(dates, price=10.0, volume=1_000_000),
        "LOW_VOL": _make_price_df(dates, price=10.0, volume=10_000),
    }
    rebalance_dates = [pd.Timestamp("2020-01-31")]

    universe = build_survivorship_aware_universe(
        panel, rebalance_dates, top_n=2, lookback_days=20, min_price=1.0
    )
    selected = set(universe["ticker"])
    assert selected == {"HIGH_VOL", "MID_VOL"}
    assert "LOW_VOL" not in selected


def test_min_price_filter_excludes_penny_stocks():
    dates = pd.date_range("2020-01-01", "2020-01-31", freq="B")
    panel = {
        "PENNY": _make_price_df(dates, price=0.50, volume=50_000_000),
        "NORMAL": _make_price_df(dates, price=20.0, volume=100_000),
    }
    rebalance_dates = [pd.Timestamp("2020-01-31")]

    universe = build_survivorship_aware_universe(
        panel, rebalance_dates, top_n=5, lookback_days=20, min_price=1.0
    )
    assert set(universe["ticker"]) == {"NORMAL"}


def test_no_future_data_used_for_ranking():
    """A ticker whose dollar volume spikes only AFTER the as_of_date must not
    be ranked using that future spike — proof the universe builder respects
    the as-of cutoff (the same core invariant the AsOfAccessor will enforce
    in M2, applied here at the universe-construction layer)."""
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    volume = pd.Series(1_000_000, index=dates)
    volume.loc[dates > pd.Timestamp("2020-01-31")] = 100_000_000  # future spike

    panel = {
        "FUTURE_SPIKE": _make_price_df(dates, price=10.0, volume=volume.values),
        "STEADY": _make_price_df(dates, price=10.0, volume=2_000_000),
    }
    rebalance_dates = [pd.Timestamp("2020-01-31")]

    universe = build_survivorship_aware_universe(
        panel, rebalance_dates, top_n=1, lookback_days=20, min_price=1.0
    )
    # As of Jan 31, STEADY has higher trailing dollar volume than FUTURE_SPIKE
    # (whose spike hasn't happened yet) — ranking must reflect that.
    assert set(universe["ticker"]) == {"STEADY"}
