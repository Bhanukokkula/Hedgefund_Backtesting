"""The candidate ticker universe, built from Tiingo's public supported_tickers.csv.

This file (https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip)
is a static, unauthenticated, unrate-limited asset listing every ticker
Tiingo has ever covered with a startDate/endDate. It substitutes for the
bulk active/delisted symbol-list endpoints FMP's free tier no longer offers
(see glassbox.data.fmp for what changed). A ticker whose endDate is not
within RECENCY_WINDOW_DAYS of the snapshot date is treated as delisted/
inactive — this is a coverage-end proxy, not an official delisting record;
glassbox.data.fmp.FMPProvider.get_delisted_symbols supplies delisting
reasons for the subset of names it covers, as a best-effort cross-check.

No hindsight: candidate selection here does not look at any ticker's
eventual fate beyond "did Tiingo's coverage end" — both active and
delisted names are drawn from the same file with the same logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "BATS"}
RECENCY_WINDOW_DAYS = 7


def is_preferred_share(ticker: str) -> bool:
    """True for Tiingo's preferred-share ticker convention ("-P-X", e.g.
    "BAC-P-W"). Preferred shares are bond-like instruments — mechanically
    low-volatility by construction and without the equity risk premium —
    not genuinely "safe" common stock. Used both when building the
    candidate universe (`load_candidate_universe`) and when filtering an
    already-cached price panel that may predate this exclusion."""
    return "-P-" in ticker


@dataclass(frozen=True)
class CandidateUniverse:
    active: pd.DataFrame  # columns: ticker, exchange, start_date, end_date
    delisted: pd.DataFrame  # columns: ticker, exchange, start_date, end_date


def load_candidate_universe(
    csv_path: Path,
    snapshot_date: date,
    recency_window_days: int = RECENCY_WINDOW_DAYS,
) -> CandidateUniverse:
    df = pd.read_csv(csv_path, dtype=str)
    df = df[(df["assetType"] == "Stock") & (df["priceCurrency"] == "USD")]
    df = df[df["exchange"].isin(US_EXCHANGES)]
    df = df.dropna(subset=["ticker"])
    # Tiingo's assetType=="Stock" does not distinguish common equity from
    # preferred shares (ticker convention "-P-X", e.g. "BAC-P-W"). Preferred
    # shares are bond-like instruments — mechanically low-volatility by
    # construction and without the equity risk premium, not "safe" common
    # stock — and this project's locked scope is common-equity factors.
    # Confirmed via the real low-vol factor result: its long decile was
    # dominated by preferred tickers and stale-priced illiquid names
    # (up to 65% zero-return days), not genuinely low-risk common stocks.
    df = df[~df["ticker"].apply(is_preferred_share)]
    df["start_date"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["endDate"], errors="coerce")
    df = df.dropna(subset=["start_date", "end_date"])
    df = df[["ticker", "exchange", "start_date", "end_date"]].drop_duplicates("ticker")

    cutoff = pd.Timestamp(snapshot_date) - pd.Timedelta(days=recency_window_days)
    active = df[df["end_date"] >= cutoff].reset_index(drop=True)
    delisted = df[df["end_date"] < cutoff].reset_index(drop=True)
    return CandidateUniverse(active=active, delisted=delisted)


def sample_tickers(
    universe: CandidateUniverse,
    n_active: int,
    n_delisted: int,
    seed: int,
    min_history_days: int = 252,
) -> pd.DataFrame:
    """Deterministically sample a working ticker set for ingestion.

    Stratifies delisted names by decade of delisting so the sample doesn't
    skew toward recent history, then filters both pools to a minimum
    lifespan so degenerate single-day listings don't dilute the factor
    library. Returns columns: ticker, exchange, start_date, end_date, is_delisted.
    """
    rng = random.Random(seed)

    def _filter_min_history(df: pd.DataFrame) -> pd.DataFrame:
        lifespan = (df["end_date"] - df["start_date"]).dt.days
        return df[lifespan >= min_history_days]

    active_pool = _filter_min_history(universe.active)
    delisted_pool = _filter_min_history(universe.delisted)

    active_sample = _seeded_sample(active_pool, n_active, rng)
    delisted_sample = _stratified_sample_by_decade(delisted_pool, n_delisted, rng)

    active_sample["is_delisted"] = False
    delisted_sample["is_delisted"] = True
    return pd.concat([active_sample, delisted_sample], ignore_index=True)


def _seeded_sample(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    n = min(n, len(df))
    idx = rng.sample(range(len(df)), n)
    return df.iloc[sorted(idx)].reset_index(drop=True).copy()


def _stratified_sample_by_decade(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    decades = (df["end_date"].dt.year // 10) * 10
    groups = df.groupby(decades)
    n_groups = len(groups)
    per_group = max(1, n // n_groups)
    parts = []
    for _, group in groups:
        k = min(per_group, len(group))
        idx = rng.sample(range(len(group)), k)
        parts.append(group.iloc[sorted(idx)])
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > n:
        idx = rng.sample(range(len(sampled)), n)
        sampled = sampled.iloc[sorted(idx)].reset_index(drop=True)
    return sampled.copy()
