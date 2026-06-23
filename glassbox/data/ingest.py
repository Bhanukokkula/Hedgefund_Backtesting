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
from glassbox.data.tiingo import TiingoProvider, TiingoRateLimitError
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


def ingest_prices(sample: pd.DataFrame, force: bool = False, sleep_seconds: float = 80.0) -> dict:
    """Pull OHLCV for each ticker in `sample`, paced well under Tiingo's
    free-tier hourly allocation (empirically ~50 requests/hour — observed a
    429 with "hourly request allocation" wording after ~50 calls in this
    project's own ingestion run). `sleep_seconds=80` caps this at 45/hour.

    Idempotent and resumable: stops cleanly (does not keep hammering) the
    moment a 429 is seen, so a later rerun with the same sample just skips
    already-cached tickers and continues from where it left off.
    """
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    provider = TiingoProvider(secrets.tiingo_api_key)
    stats = {"fetched": 0, "skipped_cached": 0, "empty": 0, "errors": 0, "rate_limited": False}

    for row in sample.itertuples():
        ticker = row.ticker
        out_path = PRICES_DIR / f"{ticker}.parquet"
        if out_path.exists() and not force:
            stats["skipped_cached"] += 1
            continue
        start = row.start_date.date()
        end = row.end_date.date()
        try:
            df = provider.get_price_history(ticker, start, end, adjusted=True)
        except TiingoRateLimitError:
            logger.warning("hourly quota exhausted; stopping cleanly, rerun later to resume")
            stats["rate_limited"] = True
            break
        except Exception:
            logger.exception("failed to fetch %s", ticker)
            stats["errors"] += 1
            continue
        if df.empty:
            stats["empty"] += 1
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
