"""Survivorship and transaction-cost sensitivity: quantify, in basis points
and Sharpe, how much each bias would have flattered the naive result.

Both functions are pure comparisons over two pre-computed return/Sharpe
pairs — they don't run a backtest themselves (glassbox.engine.backtest does
that); they exist so M5/M7 reporting always presents the delta, not just
the flattering number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glassbox.validation.metrics import sharpe_ratio


@dataclass(frozen=True)
class SurvivorshipDelta:
    full_universe_sharpe: float
    survivors_only_sharpe: float
    full_universe_annualized_return_bps: float
    survivors_only_annualized_return_bps: float
    return_inflation_bps: float
    sharpe_inflation: float


def survivorship_sensitivity(
    full_universe_returns: np.ndarray,
    survivors_only_returns: np.ndarray,
    periods_per_year: int = 252,
) -> SurvivorshipDelta:
    full_sharpe = sharpe_ratio(full_universe_returns, periods_per_year)
    surv_sharpe = sharpe_ratio(survivors_only_returns, periods_per_year)
    full_ann_return_bps = float(np.mean(full_universe_returns) * periods_per_year * 10_000)
    surv_ann_return_bps = float(np.mean(survivors_only_returns) * periods_per_year * 10_000)
    return SurvivorshipDelta(
        full_universe_sharpe=full_sharpe,
        survivors_only_sharpe=surv_sharpe,
        full_universe_annualized_return_bps=full_ann_return_bps,
        survivors_only_annualized_return_bps=surv_ann_return_bps,
        return_inflation_bps=surv_ann_return_bps - full_ann_return_bps,
        sharpe_inflation=surv_sharpe - full_sharpe,
    )


@dataclass(frozen=True)
class CostSensitivity:
    gross_sharpe: float
    net_sharpe: float
    gross_annualized_return_bps: float
    net_annualized_return_bps: float
    cost_drag_bps: float
    sharpe_drag: float


def cost_sensitivity(
    gross_returns: np.ndarray, net_returns: np.ndarray, periods_per_year: int = 252
) -> CostSensitivity:
    gross_sharpe = sharpe_ratio(gross_returns, periods_per_year)
    net_sharpe = sharpe_ratio(net_returns, periods_per_year)
    gross_ann_bps = float(np.mean(gross_returns) * periods_per_year * 10_000)
    net_ann_bps = float(np.mean(net_returns) * periods_per_year * 10_000)
    return CostSensitivity(
        gross_sharpe=gross_sharpe,
        net_sharpe=net_sharpe,
        gross_annualized_return_bps=gross_ann_bps,
        net_annualized_return_bps=net_ann_bps,
        cost_drag_bps=gross_ann_bps - net_ann_bps,
        sharpe_drag=gross_sharpe - net_sharpe,
    )
