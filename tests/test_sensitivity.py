"""Tests for survivorship and transaction-cost sensitivity calculators."""

from __future__ import annotations

import numpy as np

from glassbox.validation.sensitivity import cost_sensitivity, survivorship_sensitivity


def test_survivorship_sensitivity_flags_inflation_when_survivors_outperform():
    rng = np.random.default_rng(0)
    full_returns = rng.normal(0.0003, 0.01, 1000)
    survivors_returns = full_returns + 0.0005  # survivors look better, as expected from the bias
    delta = survivorship_sensitivity(full_returns, survivors_returns)
    assert delta.return_inflation_bps > 0
    assert delta.survivors_only_sharpe > delta.full_universe_sharpe


def test_survivorship_sensitivity_zero_when_identical():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0002, 0.01, 500)
    delta = survivorship_sensitivity(returns, returns)
    assert delta.return_inflation_bps == 0.0
    assert delta.sharpe_inflation == 0.0


def test_cost_sensitivity_shows_drag_when_net_is_worse():
    rng = np.random.default_rng(2)
    gross_returns = rng.normal(0.0005, 0.01, 1000)
    net_returns = gross_returns - 0.0002  # costs eat into every period's return
    sens = cost_sensitivity(gross_returns, net_returns)
    assert sens.cost_drag_bps > 0
    assert sens.gross_sharpe > sens.net_sharpe


def test_cost_sensitivity_zero_drag_when_costless():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0003, 0.01, 500)
    sens = cost_sensitivity(returns, returns)
    assert sens.cost_drag_bps == 0.0
    assert sens.sharpe_drag == 0.0
