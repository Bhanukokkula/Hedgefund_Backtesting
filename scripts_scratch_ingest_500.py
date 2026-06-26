"""One-off driver: pull the full 500-ticker target sample, resuming across
Tiingo's hourly quota resets automatically. Not part of the package API —
a scratch script for this long-running operational pull."""

import logging
import time
from datetime import date

from glassbox.data.ingest import build_sample, ingest_prices
from glassbox.data.tiingo import TiingoProvider, TiingoRateLimitError
from glassbox.settings import secrets, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_500")

N_ACTIVE = 250
N_DELISTED = 250
QUOTA_POLL_SECONDS = 180

sample = build_sample(N_ACTIVE, N_DELISTED, settings.seed, date.today())
logger.info(
    "target sample: %d active, %d delisted", (~sample.is_delisted).sum(), sample.is_delisted.sum()
)

probe = TiingoProvider(secrets.tiingo_api_key)

while True:
    stats = ingest_prices(sample, force=False, sleep_seconds=80.0)
    logger.info("batch stats: %s", stats)
    if not stats["rate_limited"]:
        logger.info("ingestion complete, no more rate limiting")
        break
    logger.info("waiting for Tiingo hourly quota to reset...")
    while True:
        time.sleep(QUOTA_POLL_SECONDS)
        try:
            probe.get_price_history("MSFT", date(2024, 1, 1), date(2024, 1, 5), adjusted=True)
            break
        except TiingoRateLimitError:
            continue

logger.info("DONE")
