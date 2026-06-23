"""Decile ranking and portfolio construction from cross-sectional factor scores.

Higher score = more attractive (long side); lower score = less attractive
(short side) — see glassbox.factors.scoring for how each factor's score is
oriented so this convention holds uniformly.
"""

from __future__ import annotations

import pandas as pd


def decile_rank(scores: dict[str, float], n_deciles: int = 10) -> dict[str, int]:
    """Rank tickers into deciles 1 (lowest score) .. n_deciles (highest score).

    Uses rank-based binning so ties and small samples degrade gracefully
    rather than raising on duplicate bin edges (pd.qcut's default failure
    mode with few unique scores).
    """
    if not scores:
        return {}
    series = pd.Series(scores)
    # Drop non-finite scores (NaN/inf) before ranking — a single bad score
    # would otherwise propagate a NaN decile and crash the int cast below,
    # silently dropping every other ticker's rank along with it.
    series = series[series.apply(lambda x: pd.notna(x) and abs(x) != float("inf"))]
    if series.empty:
        return {}
    ranks = series.rank(method="first")
    deciles = pd.qcut(ranks, q=min(n_deciles, len(series)), labels=False, duplicates="drop") + 1
    return deciles.astype(int).to_dict()


def decile_long_short_weights(scores: dict[str, float], n_deciles: int = 10) -> dict[str, float]:
    """Equal-weight long the top decile, equal-weight short the bottom decile.

    Gross exposure is 100% long / 100% short (dollar-neutral); weights sum
    to 0.0 across the whole book.
    """
    deciles = decile_rank(scores, n_deciles)
    if not deciles:
        return {}
    top = max(deciles.values())
    bottom = min(deciles.values())
    long_tickers = [t for t, d in deciles.items() if d == top]
    short_tickers = [t for t, d in deciles.items() if d == bottom]

    weights = {}
    if long_tickers:
        w = 1.0 / len(long_tickers)
        for t in long_tickers:
            weights[t] = w
    if short_tickers:
        w = -1.0 / len(short_tickers)
        for t in short_tickers:
            weights[t] = weights.get(t, 0.0) + w
    return weights


def long_only_top_decile_weights(scores: dict[str, float], n_deciles: int = 10) -> dict[str, float]:
    deciles = decile_rank(scores, n_deciles)
    if not deciles:
        return {}
    top = max(deciles.values())
    long_tickers = [t for t, d in deciles.items() if d == top]
    if not long_tickers:
        return {}
    w = 1.0 / len(long_tickers)
    return dict.fromkeys(long_tickers, w)


def decile_mean_scores(scores: dict[str, float], n_deciles: int = 10) -> dict[int, float]:
    """Mean raw score per decile — used to check monotonicity (M4 accept
    criteria: each factor produces a decile spread with sensible
    monotonicity)."""
    deciles = decile_rank(scores, n_deciles)
    if not deciles:
        return {}
    df = pd.DataFrame({"score": scores, "decile": pd.Series(deciles)})
    return df.groupby("decile")["score"].mean().to_dict()
