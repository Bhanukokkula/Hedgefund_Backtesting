"""Tests for the Tiingo-supported_tickers.csv-based candidate universe.

Uses a small synthetic CSV fixture, not the network, per the project's rule
that engine/data-layer logic must be testable without live API calls.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from glassbox.data.candidates import load_candidate_universe, sample_tickers

SYNTHETIC_CSV = """ticker,exchange,assetType,priceCurrency,startDate,endDate
AAA,NASDAQ,Stock,USD,2000-01-01,2026-06-22
BBB,NYSE,Stock,USD,2001-01-01,2026-06-20
CCC,NASDAQ,Stock,USD,2005-01-01,2010-12-31
DDD,NYSE,Stock,USD,2002-01-01,2015-06-30
EEE,AMEX,Stock,USD,2010-01-01,2026-06-22
FFF,NASDAQ,ETF,USD,2010-01-01,2026-06-22
GGG,SHE,Stock,CNY,2010-01-01,2026-06-22
HHH,NASDAQ,Stock,USD,2020-01-01,2020-02-01
III,NYSE,Stock,USD,1998-01-01,2008-12-31
BAC-P-W,NYSE,Stock,USD,2010-01-01,2026-06-22
SF-P-C,NYSE,Stock,USD,2010-01-01,2010-12-31
"""


def _write_csv(tmp_path):
    path = tmp_path / "tickers.csv"
    path.write_text(SYNTHETIC_CSV)
    return path


def test_active_vs_delisted_split(tmp_path):
    path = _write_csv(tmp_path)
    snapshot = date(2026, 6, 22)
    universe = load_candidate_universe(path, snapshot)

    active_tickers = set(universe.active["ticker"])
    delisted_tickers = set(universe.delisted["ticker"])

    assert active_tickers == {"AAA", "BBB", "EEE"}
    assert delisted_tickers == {"CCC", "DDD", "HHH", "III"}


def test_non_us_and_non_stock_excluded(tmp_path):
    path = _write_csv(tmp_path)
    universe = load_candidate_universe(path, date(2026, 6, 22))
    all_tickers = set(universe.active["ticker"]) | set(universe.delisted["ticker"])
    assert "FFF" not in all_tickers  # ETF, not Stock
    assert "GGG" not in all_tickers  # non-US exchange / non-USD


def test_load_candidate_universe_handles_missing_ticker_values(tmp_path):
    """Regression test: a blank ticker field (real rows in Tiingo's CSV)
    must not crash the preferred-share filter — is_preferred_share() can
    only run on string values, so the NaN-ticker rows must be dropped
    first. An earlier version of this function applied the filter before
    dropna(subset=["ticker"]) and crashed with TypeError on the first real
    NaN ticker it hit."""
    csv_with_blank_ticker = SYNTHETIC_CSV + ",NASDAQ,Stock,USD,2010-01-01,2026-06-22\n"
    path = tmp_path / "tickers.csv"
    path.write_text(csv_with_blank_ticker)
    universe = load_candidate_universe(path, date(2026, 6, 22))
    assert "AAA" in set(universe.active["ticker"])


def test_preferred_shares_excluded(tmp_path):
    """Regression test: preferred shares (ticker convention "-P-X") are
    bond-like instruments without the equity risk premium, mechanically
    low-volatility by construction — not genuinely "safe" common stock.
    Confirmed via the real low-vol factor backtest: its long decile was
    dominated by preferred tickers before this filter existed."""
    path = _write_csv(tmp_path)
    universe = load_candidate_universe(path, date(2026, 6, 22))
    all_tickers = set(universe.active["ticker"]) | set(universe.delisted["ticker"])
    assert "BAC-P-W" not in all_tickers
    assert "SF-P-C" not in all_tickers


def test_no_hindsight_in_classification(tmp_path):
    """Active/delisted classification depends only on endDate vs snapshot_date,
    never on any later-known fact — i.e. the same row classified at an
    earlier snapshot date should be 'active' even though it is 'delisted' as
    of today."""
    path = _write_csv(tmp_path)
    universe_2008 = load_candidate_universe(path, date(2008, 6, 1))
    # DDD doesn't delist until 2015-06-30, so as of 2008 it must show as active.
    assert "DDD" in set(universe_2008.active["ticker"])
    assert "DDD" not in set(universe_2008.delisted["ticker"])


def test_sample_respects_min_history_days(tmp_path):
    path = _write_csv(tmp_path)
    universe = load_candidate_universe(path, date(2026, 6, 22))
    # HHH has only a 31-day lifespan; with a 252-day floor it must be excluded.
    sample = sample_tickers(universe, n_active=10, n_delisted=10, seed=1, min_history_days=252)
    assert "HHH" not in set(sample["ticker"])


def test_sample_is_deterministic_given_seed(tmp_path):
    path = _write_csv(tmp_path)
    universe = load_candidate_universe(path, date(2026, 6, 22))
    sample_a = sample_tickers(universe, n_active=2, n_delisted=2, seed=42, min_history_days=0)
    sample_b = sample_tickers(universe, n_active=2, n_delisted=2, seed=42, min_history_days=0)
    pd.testing.assert_frame_equal(sample_a.reset_index(drop=True), sample_b.reset_index(drop=True))


def test_sample_marks_delisted_flag_correctly(tmp_path):
    path = _write_csv(tmp_path)
    universe = load_candidate_universe(path, date(2026, 6, 22))
    sample = sample_tickers(universe, n_active=10, n_delisted=10, seed=7, min_history_days=0)
    delisted_in_sample = set(sample.loc[sample["is_delisted"], "ticker"])
    active_in_sample = set(sample.loc[~sample["is_delisted"], "ticker"])
    assert delisted_in_sample <= {"CCC", "DDD", "HHH", "III"}
    assert active_in_sample <= {"AAA", "BBB", "EEE"}
