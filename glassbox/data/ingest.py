"""M1 ingestion orchestration: the only script in this repo that touches the network.

Run as: .venv/bin/python -m glassbox.data.ingest

Pulls a deterministic sample of active + delisted US-exchange common stocks
(see glassbox.data.candidates), fetches full daily OHLCV from Tiingo for
each, and writes one parquet file per ticker. Idempotent: tickers already
cached on disk are skipped on rerun unless --force is passed.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

from glassbox.data.candidates import load_candidate_universe, sample_tickers
from glassbox.data.tiingo import TiingoMonthlyQuotaError, TiingoProvider, TiingoRateLimitError
from glassbox.settings import secrets, settings

logger = logging.getLogger(__name__)

SUPPORTED_TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
RAW_TICKERS_CSV = settings.raw_dir / "tiingo_supported_tickers.csv"
PRICES_DIR = settings.parquet_dir / "prices"
CANDIDATE_SAMPLE_PATH = settings.parquet_dir / "candidate_sample.parquet"


def ensure_supported_tickers_csv() -> Path:
    if RAW_TICKERS_CSV.exists():
        return RAW_TICKERS_CSV
    import io
    import zipfile

    import requests

    RAW_TICKERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SUPPORTED_TICKERS_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as src, open(RAW_TICKERS_CSV, "wb") as dst:
            dst.write(src.read())
    return RAW_TICKERS_CSV


def build_sample(n_active: int, n_delisted: int, seed: int, snapshot_date: date) -> pd.DataFrame:
    csv_path = ensure_supported_tickers_csv()
    universe = load_candidate_universe(csv_path, snapshot_date)
    sample = sample_tickers(universe, n_active=n_active, n_delisted=n_delisted, seed=seed)
    CANDIDATE_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(CANDIDATE_SAMPLE_PATH, index=False)
    return sample


KNOWN_BAD_PATH = PRICES_DIR / "_known_bad.txt"


def _load_known_bad() -> set[str]:
    if not KNOWN_BAD_PATH.exists():
        return set()
    return set(KNOWN_BAD_PATH.read_text().splitlines())


def _append_known_bad(ticker: str) -> None:
    KNOWN_BAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KNOWN_BAD_PATH, "a") as f:
        f.write(f"{ticker}\n")


def ingest_prices(sample: pd.DataFrame, force: bool = False, sleep_seconds: float = 80.0) -> dict:
    """Pull OHLCV for each ticker in `sample`, paced well under Tiingo's
    free-tier hourly allocation (empirically ~50 requests/hour — observed a
    429 with "hourly request allocation" wording after ~50 calls in this
    project's own ingestion run). `sleep_seconds=80` caps this at 45/hour.

    Idempotent and resumable: stops cleanly (does not keep hammering) the
    moment a 429 is seen, so a later rerun with the same sample just skips
    already-cached tickers and continues from where it left off. Every
    request paces equally regardless of outcome — an empty/error response
    still counts against Tiingo's hourly allocation, so skipping the sleep
    on failure (as an earlier version of this function did) just burns
    quota faster on the doomed tickers and leaves less for ones that would
    actually succeed. Tickers that error or come back empty are recorded
    in `_known_bad.txt` so they are skipped (not retried from scratch) on
    every future resume — Tiingo genuinely not having data for an obscure
    SPAC warrant is a permanent fact, not a transient failure.
    """
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    provider = TiingoProvider(secrets.tiingo_api_key)
    known_bad = _load_known_bad()
    stats = {
        "fetched": 0,
        "skipped_cached": 0,
        "skipped_known_bad": 0,
        "empty": 0,
        "errors": 0,
        "rate_limited": False,
        "monthly_quota_exhausted": False,
    }

    for row in sample.itertuples():
        ticker = row.ticker
        out_path = PRICES_DIR / f"{ticker}.parquet"
        if out_path.exists() and not force:
            stats["skipped_cached"] += 1
            continue
        if ticker in known_bad and not force:
            stats["skipped_known_bad"] += 1
            continue
        start = row.start_date.date()
        end = row.end_date.date()
        try:
            df = provider.get_price_history(ticker, start, end, adjusted=True)
        except TiingoRateLimitError:
            logger.warning("hourly quota exhausted; stopping cleanly, rerun later to resume")
            stats["rate_limited"] = True
            break
        except TiingoMonthlyQuotaError:
            # A hard wall, not a per-ticker fact: do NOT blacklist `ticker`,
            # it almost certainly has perfectly good data — Tiingo just
            # won't let us look up any NEW unique symbol until next month
            # (or a paid upgrade). Stop the whole run; retrying later this
            # month will hit the same wall immediately.
            logger.warning(
                "monthly unique-symbol quota exhausted on %s; stopping, will not "
                "resolve until next month (or a paid plan) — not treating %s as bad data",
                ticker,
                ticker,
            )
            stats["monthly_quota_exhausted"] = True
            break
        except Exception:
            logger.exception("failed to fetch %s", ticker)
            stats["errors"] += 1
            _append_known_bad(ticker)
            time.sleep(sleep_seconds)
            continue
        if df.empty:
            stats["empty"] += 1
            _append_known_bad(ticker)
            time.sleep(sleep_seconds)
            continue
        df.to_parquet(out_path, index=False)
        stats["fetched"] += 1
        logger.info("fetched %s (%d rows)", ticker, len(df))
        time.sleep(sleep_seconds)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-active", type=int, default=60)
    parser.add_argument("--n-delisted", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    snapshot_date = date.today()
    sample = build_sample(args.n_active, args.n_delisted, settings.seed, snapshot_date)
    logger.info(
        "sample built: %d active, %d delisted",
        (~sample.is_delisted).sum(),
        sample.is_delisted.sum(),
    )

    stats = ingest_prices(sample, force=args.force)
    logger.info("ingestion done: %s", stats)


if __name__ == "__main__":
    main()
