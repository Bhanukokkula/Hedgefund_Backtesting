"""As-of-correct corporate-action adjustment.

Tiingo's `adjClose` bakes in every split and dividend known as of the data
pull date — including ones that happen AFTER any historical date you might
query. Handing that column to strategy code for a backtest as-of some past
date would be a silent look-ahead: the strategy would see a price series
already adjusted for a split that, as of that historical date, had not
happened yet.

This module reconstructs the adjustment using only rows dated <= as_of_date,
via the standard backward-recursive method (also how Tiingo computes
adjClose, just truncated to the as-of window instead of "through today"):

    factor[last] = 1.0
    factor[i] = factor[i+1] / split_factor[i+1] * (1 - div_cash[i+1] / close[i+1])
    adj_close[i] = close[i] * factor[i]

`split_factor[i+1]` and `div_cash[i+1]` are the action recorded on the day
immediately after day i — i.e. the action that took effect between day i
and day i+1. Dividing by split_factor (rather than multiplying) is what
makes a pre-split price comparable to a post-split one: a 2-for-1 split
roughly halves the price, so the pre-split price must be halved too to
sit on the same continuous series. See tests/test_adjustments.py for a
hand-checked split case and a hand-checked dividend case.
"""

from __future__ import annotations

import pandas as pd


def reconstruct_asof_adjusted_close(df: pd.DataFrame) -> pd.Series:
    """`df` must be sorted ascending by date, already truncated to
    date <= as_of_date, with columns: close, split_factor, div_cash.

    Returns a Series of as-of-correct adjusted close, indexed the same as df.
    The last row's adjusted close always equals its raw close (factor=1.0)
    by construction — there is nothing after as_of_date to adjust for.
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    close = df["close"].to_numpy()
    split_factor = df["split_factor"].to_numpy()
    div_cash = df["div_cash"].to_numpy()

    factor = [1.0] * n
    for i in range(n - 2, -1, -1):
        next_split = split_factor[i + 1] if split_factor[i + 1] else 1.0
        next_close = close[i + 1]
        div_adj = 1.0 - (div_cash[i + 1] / next_close) if next_close else 1.0
        factor[i] = factor[i + 1] / next_split * div_adj

    adj_close = close * factor
    return pd.Series(adj_close, index=df.index, name="asof_adj_close")
