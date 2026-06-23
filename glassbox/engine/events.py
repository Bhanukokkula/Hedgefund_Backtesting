"""Event types for the single monotonic simulation clock.

The engine processes one trading date at a time, in this fixed order per
date: MarketOpen (pending fills execute here) -> ... -> Rebalance (signal
generated here, using that date's close) -> MarketClose (NAV marked here).
Orders generated at a Rebalance event are queued and can only fill at the
following date's MarketOpen — never the same date's close. That ordering is
what test_no_same_bar_fill in tests/test_backtest.py exists to prove.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class MarketOpen:
    as_of_date: date


@dataclass(frozen=True)
class Rebalance:
    as_of_date: date
    target_weights: dict[str, float]


@dataclass(frozen=True)
class MarketClose:
    as_of_date: date
