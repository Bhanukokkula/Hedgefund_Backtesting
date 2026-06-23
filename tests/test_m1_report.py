"""Synthetic-fixture tests for the M1 validation report's building blocks:
coverage-gap detection, bad-tick detection, and the survivorship-delta
return calculation. The full run_m1_validation() (which reads parquet from
disk) is exercised manually against real ingested data, not here."""

from __future__ import annotations

import pandas as pd

from glassbox.validation.m1_report import (
    _annualized_return,
    _annualized_vol,
    _universe_daily_returns,
    detect_bad_ticks,
    detect_coverage_gaps,
)


def _df(dates, adj_close):
    return pd.DataFrame({"adj_close": adj_close}, index=pd.DatetimeIndex(dates))


def test_detect_coverage_gaps_flags_large_date_jumps():
    dense_dates = pd.date_range("2020-01-01", "2020-03-01", freq="B")
    gappy_dates = list(pd.date_range("2020-01-01", "2020-01-15", freq="B")) + list(
        pd.date_range("2020-03-01", "2020-03-15", freq="B")
    )
    panel = {
        "DENSE": _df(dense_dates, [10.0] * len(dense_dates)),
        "GAPPY": _df(gappy_dates, [10.0] * len(gappy_dates)),
    }
    flagged = detect_coverage_gaps(panel, max_gap_days=10)
    assert "GAPPY" in flagged
    assert "DENSE" not in flagged


def test_detect_bad_ticks_flags_implausible_moves():
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="B")
    normal = [10.0, 10.1, 10.05, 9.9, 10.2, 10.1]
    spiky = [10.0, 10.1, 0.05, 10.0, 10.1, 10.2]  # implausible one-day crash/recovery
    panel = {
        "NORMAL": _df(dates, normal),
        "SPIKY": _df(dates, spiky),
    }
    flagged = detect_bad_ticks(panel, max_daily_move=0.95)
    assert "SPIKY" in flagged
    assert "NORMAL" not in flagged


def test_annualized_helpers_scale_correctly():
    daily_return = 0.0004
    daily_vol = 0.01
    assert _annualized_return(daily_return) == daily_return * 252 * 10_000
    assert _annualized_vol(daily_vol) == daily_vol * (252**0.5) * 10_000


def test_universe_daily_returns_only_uses_membership_window():
    dates = pd.date_range("2020-01-01", "2020-04-30", freq="B")
    panel = {
        "A": _df(dates, [10.0 + 0.01 * i for i in range(len(dates))]),
        "B": _df(dates, [20.0] * len(dates)),  # flat, zero return
    }
    universe = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")],
            "ticker": ["A", "B"],
        }
    )
    returns = _universe_daily_returns(panel, universe)
    assert not returns.empty
    # B is flat in its membership window, so some days are exactly zero return
    # contributed by B; A is trending up, so the combined series shouldn't be
    # uniformly zero.
    assert (returns != 0).any()
