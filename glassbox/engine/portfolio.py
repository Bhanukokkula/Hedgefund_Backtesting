"""Portfolio accounting: positions (shares), cash, NAV, turnover.

Deliberately dumb and auditable — no hidden margin, no implicit leverage.
Every dollar of cost is deducted from cash at the moment a trade fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    total_cost_paid: float = 0.0
    total_traded_value: float = 0.0

    def position_value(self, prices: dict[str, float]) -> float:
        return sum(
            shares * prices[ticker] for ticker, shares in self.positions.items() if ticker in prices
        )

    def nav(self, prices: dict[str, float]) -> float:
        return self.cash + self.position_value(prices)

    def apply_trade(self, ticker: str, shares_delta: float, fill_price: float, cost: float) -> None:
        trade_value = shares_delta * fill_price
        self.cash -= trade_value
        self.cash -= cost
        self.total_cost_paid += cost
        self.total_traded_value += abs(trade_value)
        self.positions[ticker] = self.positions.get(ticker, 0.0) + shares_delta
        if abs(self.positions[ticker]) < 1e-12:
            del self.positions[ticker]
