"""Cross-sectional, price-derived factor scores — all computed through an
AsOfAccessor, so no factor can see a price dated after its as_of_date.

Momentum (12-1): trailing 12-month return, skipping the most recent month
(the standard Jegadeesh-Titman construction — skipping the last month avoids
conflating momentum with short-term reversal).

Short-term reversal: negative of the most recent 1-month return — past
losers score high (go long), past winners score low (go short).

Low-volatility: negative of trailing realized volatility — low-vol names
score high (go long), matching the low-volatility anomaly's long side.
"""

from __future__ import annotations

import numpy as np

from glassbox.engine.asof_accessor import ConcreteAsOfAccessor

TRADING_DAYS_PER_MONTH = 21


def momentum_score(
    accessor: ConcreteAsOfAccessor,
    tickers: list[str],
    lookback_months: int,
    skip_months: int,
) -> dict[str, float]:
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = skip_months * TRADING_DAYS_PER_MONTH
    scores = {}
    for ticker in tickers:
        series = accessor.price_series(ticker, lookback_days=lookback_days, adjusted=True)
        if len(series) <= skip_days + 1:
            continue
        start_price = series.iloc[0]
        end_price = series.iloc[-1 - skip_days]
        if start_price <= 0:
            continue
        scores[ticker] = (end_price / start_price) - 1.0
    return scores


def reversal_score(
    accessor: ConcreteAsOfAccessor,
    tickers: list[str],
    lookback_months: int,
) -> dict[str, float]:
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    scores = {}
    for ticker in tickers:
        series = accessor.price_series(ticker, lookback_days=lookback_days, adjusted=True)
        if len(series) < 2:
            continue
        start_price, end_price = series.iloc[0], series.iloc[-1]
        if start_price <= 0:
            continue
        one_month_return = (end_price / start_price) - 1.0
        scores[ticker] = -one_month_return
    return scores


def low_vol_score(
    accessor: ConcreteAsOfAccessor,
    tickers: list[str],
    lookback_days: int,
) -> dict[str, float]:
    scores = {}
    for ticker in tickers:
        series = accessor.price_series(ticker, lookback_days=lookback_days, adjusted=True)
        if len(series) < 2:
            continue
        returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            continue
        vol = float(np.std(returns, ddof=1))
        if not np.isfinite(vol) or vol <= 0:
            continue
        scores[ticker] = -vol
    return scores
