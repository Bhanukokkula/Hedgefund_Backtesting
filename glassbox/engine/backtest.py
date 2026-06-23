"""The event-driven backtest engine: one monotonic clock, no same-bar fills.

Per trading date, processing order is fixed:
  1. MarketOpen — any orders queued from the PREVIOUS date's rebalance
     signal fill here, at this date's open price.
  2. Rebalance  — if this date has a signal, target weights are computed
     using THIS date's close price and queued as orders for the NEXT
     date's open. They never fill against today's own close.
  3. MarketClose — NAV is marked using this date's close price.

This ordering is the whole point of M3: a strategy cannot generate a signal
from today's close and trade at today's close — see
tests/test_backtest.py::test_no_same_bar_fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from glassbox.engine.costs import CostModel
from glassbox.engine.portfolio import Portfolio


@dataclass
class NavRecord:
    as_of_date: pd.Timestamp
    nav: float


class BacktestEngine:
    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        trading_dates: list[pd.Timestamp],
        rebalance_schedule: dict[pd.Timestamp, dict[str, float]],
        cost_model: CostModel,
        initial_cash: float,
    ):
        self.panel = panel
        self.trading_dates = trading_dates
        self.rebalance_schedule = rebalance_schedule
        self.cost_model = cost_model
        self.portfolio = Portfolio(cash=initial_cash)
        self.nav_history: list[NavRecord] = []
        self._pending_orders: dict[str, float] | None = None

    def run(self) -> list[NavRecord]:
        for as_of_date in self.trading_dates:
            prices_open = self._prices_for(as_of_date, "open")
            prices_close = self._prices_for(as_of_date, "close")

            if self._pending_orders:
                self._execute_fills(as_of_date, prices_open)
                self._pending_orders = None

            if as_of_date in self.rebalance_schedule:
                self._queue_orders(as_of_date, prices_close)

            nav = self.portfolio.nav(prices_close)
            self.nav_history.append(NavRecord(as_of_date, nav))

        return self.nav_history

    def _prices_for(self, as_of_date: pd.Timestamp, field: str) -> dict[str, float]:
        prices = {}
        for ticker, df in self.panel.items():
            if as_of_date in df.index:
                prices[ticker] = float(df.loc[as_of_date, field])
        return prices

    def _execute_fills(self, as_of_date: pd.Timestamp, prices_open: dict[str, float]) -> None:
        for ticker, shares_delta in self._pending_orders.items():
            if ticker not in prices_open:
                continue
            fill_price = prices_open[ticker]
            trade_value = shares_delta * fill_price
            volume = self._volume_for(ticker, as_of_date)
            participation_rate = abs(shares_delta) / volume if volume else 0.0
            cost = self.cost_model.trade_cost(trade_value, participation_rate)
            self.portfolio.apply_trade(ticker, shares_delta, fill_price, cost)

    def _volume_for(self, ticker: str, as_of_date: pd.Timestamp) -> float | None:
        df = self.panel.get(ticker)
        if df is None or "volume" not in df.columns or as_of_date not in df.index:
            return None
        return float(df.loc[as_of_date, "volume"])

    def _queue_orders(self, as_of_date: pd.Timestamp, prices_close: dict[str, float]) -> None:
        target_weights = self.rebalance_schedule[as_of_date]
        current_nav = self.portfolio.nav(prices_close)
        target_shares = {
            ticker: (weight * current_nav) / prices_close[ticker]
            for ticker, weight in target_weights.items()
            if ticker in prices_close
        }
        all_tickers = set(self.portfolio.positions) | set(target_shares)
        new_pending = {}
        for ticker in all_tickers:
            current_shares = self.portfolio.positions.get(ticker, 0.0)
            desired_shares = target_shares.get(ticker, 0.0)
            delta = desired_shares - current_shares
            if abs(delta) > 1e-9:
                new_pending[ticker] = delta
        self._pending_orders = new_pending
