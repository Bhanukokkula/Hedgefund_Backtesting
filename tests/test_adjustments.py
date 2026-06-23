"""Hand-checked corporate-action reconstruction tests (M2 acceptance criteria)."""

from __future__ import annotations

import pandas as pd

from glassbox.engine.adjustments import reconstruct_asof_adjusted_close


def test_two_for_one_split_hand_check():
    # No real price movement, just a 2-for-1 split on the last day.
    # Pre-split $100 days must reconstruct to the same $50 level as the
    # post-split day, since nothing actually changed in shareholder value.
    df = pd.DataFrame(
        {
            "close": [100.0, 100.0, 50.0],
            "split_factor": [1.0, 1.0, 2.0],
            "div_cash": [0.0, 0.0, 0.0],
        }
    )
    adj = reconstruct_asof_adjusted_close(df)
    assert adj.iloc[2] == 50.0  # as_of day: always equals raw close
    assert adj.iloc[1] == 50.0
    assert adj.iloc[0] == 50.0


def test_cash_dividend_hand_check():
    # $1 dividend paid on the last day, price drops from 100 to 99 (the
    # mechanical ex-dividend drop). The day before should adjust down by
    # the same proportion so the series reflects total return.
    df = pd.DataFrame(
        {
            "close": [100.0, 99.0],
            "split_factor": [1.0, 1.0],
            "div_cash": [0.0, 1.0],
        }
    )
    adj = reconstruct_asof_adjusted_close(df)
    assert adj.iloc[1] == 99.0
    expected_day0 = 100.0 * (1 - 1.0 / 99.0)
    assert abs(adj.iloc[0] - expected_day0) < 1e-9


def test_as_of_day_always_equals_raw_close():
    df = pd.DataFrame(
        {
            "close": [10.0, 20.0, 30.0, 40.0],
            "split_factor": [1.0, 3.0, 1.0, 1.0],
            "div_cash": [0.0, 0.0, 0.5, 0.0],
        }
    )
    adj = reconstruct_asof_adjusted_close(df)
    assert adj.iloc[-1] == df["close"].iloc[-1]


def test_truncating_series_does_not_change_earlier_adjusted_values():
    """The core invariant applied to adjustments: reconstructing as-of an
    earlier date must not be affected by rows that come after it. Adjusting
    through day 2 must give the same day-0 and day-1 values as adjusting
    through day 1 alone, when no corporate action occurs between them."""
    full = pd.DataFrame(
        {
            "close": [50.0, 55.0, 60.0],
            "split_factor": [1.0, 1.0, 2.0],
            "div_cash": [0.0, 0.0, 0.0],
        }
    )
    truncated = full.iloc[:2]

    adj_full = reconstruct_asof_adjusted_close(full)
    adj_truncated = reconstruct_asof_adjusted_close(truncated)

    assert adj_truncated.iloc[0] == 50.0
    assert adj_truncated.iloc[1] == 55.0
    # In the full series, day 2's split retroactively changes days 0/1's
    # adjusted values relative to day 2 — that's expected and correct
    # (it's why an as-of view must truncate before adjusting, not after).
    assert adj_full.iloc[0] != adj_truncated.iloc[0]
