"""Adversarial tests: deliberately try to read future data through
ConcreteAsOfAccessor and assert it is refused/truncated, not silently
served. This is what makes the core invariant real rather than aspirational:

    No computation may read any datum whose knowable-date is later than the
    simulation's current as-of clock.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from glassbox.engine.asof_accessor import ConcreteAsOfAccessor


def _panel_with_future_spike():
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    close = pd.Series(10.0, index=dates)
    close.loc[dates > pd.Timestamp("2020-02-14")] = 9999.0  # an obvious future spike
    df = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1_000_000,
            "split_factor": 1.0,
            "div_cash": 0.0,
        },
        index=dates,
    )
    return {"SPY": df}


def test_price_series_never_returns_rows_after_as_of():
    panel = _panel_with_future_spike()
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 2, 14), panel=panel)
    series = accessor.price_series("SPY", lookback_days=60, adjusted=False)
    assert (series == 9999.0).sum() == 0
    assert series.index.max() <= pd.Timestamp("2020-02-14")


def test_latest_price_does_not_see_tomorrows_spike():
    panel = _panel_with_future_spike()
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 2, 14), panel=panel)
    assert accessor.latest_price("SPY") == 10.0  # not 9999.0


def test_adjusted_price_series_excludes_future_split():
    """A split recorded AFTER as_of_date must not retroactively change the
    as-of-correct adjusted series — i.e. you cannot smuggle a future split
    into a historical view by asking for adjusted=True."""
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
    close = pd.Series([100.0] * 6 + [50.0] * 2, index=dates)
    split_factor = pd.Series(1.0, index=dates)
    split_factor.loc[dates[6]] = 2.0  # split happens AFTER day 5 (index of as_of below)
    df = pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1_000_000,
            "split_factor": split_factor,
            "div_cash": 0.0,
        },
        index=dates,
    )
    panel = {"AAA": df}
    as_of = dates[5]  # day before the split
    accessor = ConcreteAsOfAccessor(as_of_date=as_of.date(), panel=panel)
    series = accessor.price_series("AAA", lookback_days=10, adjusted=True)
    # As of day 5, no split has happened yet — every day must show $100, not
    # a retroactively-split-adjusted $50.
    assert (series == 100.0).all()


def test_advance_to_rejects_moving_clock_backward():
    panel = _panel_with_future_spike()
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 2, 14), panel=panel)
    with pytest.raises(ValueError):
        accessor.advance_to(date(2020, 1, 1))


def test_advance_to_forward_then_sees_new_data():
    panel = _panel_with_future_spike()
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 2, 14), panel=panel)
    later = accessor.advance_to(date(2020, 3, 1))
    assert later.latest_price("SPY") == 9999.0
    # the original accessor instance is unmutated (immutable, frozen dataclass)
    assert accessor.latest_price("SPY") == 10.0


def test_universe_membership_truncated_to_as_of():
    universe_table = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-06-30")],
            "ticker": ["JAN_NAME", "JUN_NAME"],
        }
    )
    accessor = ConcreteAsOfAccessor(
        as_of_date=date(2020, 2, 1), panel={}, universe_table=universe_table
    )
    assert accessor.universe() == ["JAN_NAME"]  # cannot see the June rebalance yet


def test_is_tradable_false_once_ticker_has_no_recent_data():
    dates = pd.date_range("2020-01-01", "2020-01-31", freq="B")
    df = pd.DataFrame(
        {
            "close": 10.0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "volume": 1000,
            "split_factor": 1.0,
            "div_cash": 0.0,
        },
        index=dates,
    )
    panel = {"DELISTED": df}
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 6, 1), panel=panel)
    assert accessor.is_tradable("DELISTED") is False


def test_shares_outstanding_missing_returns_none_not_a_guess():
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 1, 1), panel={})
    assert accessor.shares_outstanding("ANY") is None
    assert accessor.market_cap("ANY") is None
