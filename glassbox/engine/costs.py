"""Transaction cost model: commission + slippage (half-spread + market impact).

All three components are configurable (config.yaml: costs.*) and apply
symmetrically to buys and sells — there is no asymmetry that could be used
to flatter a strategy's net returns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float
    half_spread_bps: float
    market_impact_coefficient: float
    participation_rate_cap: float

    def trade_cost(self, trade_value: float, participation_rate: float) -> float:
        """Cost in dollars for a trade of `trade_value` (signed or unsigned;
        only magnitude matters) at the given `participation_rate`
        (trade shares / average daily volume, already non-negative)."""
        capped_participation = min(participation_rate, self.participation_rate_cap)
        commission = abs(trade_value) * self.commission_bps / 10_000
        slippage = abs(trade_value) * (
            self.half_spread_bps / 10_000 + self.market_impact_coefficient * capped_participation
        )
        return commission + slippage
