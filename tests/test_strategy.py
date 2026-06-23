"""M6 acceptance test: a strategy is data (StrategySpec), and running the
same spec against the same panel twice — once as if from the CLI, once as
if from the dashboard — produces byte-identical results, because both call
the same run_strategy() function."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glassbox.strategy import StrategySpec, run_strategy


def _make_panel(n_tickers=20, n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    panel = {}
    for i in range(n_tickers):
        drift = rng.uniform(-0.001, 0.002)
        close = 50.0 * (1.0 + drift) ** np.arange(n_days)
        close = close * (1 + rng.normal(0, 0.01, n_days)).cumprod()
        close = np.abs(close) + 1.0
        panel[f"T{i}"] = pd.DataFrame(
            {
                "close": close,
                "adj_close": close,
                "open": close,
                "high": close,
                "low": close,
                "volume": rng.integers(100_000, 5_000_000, n_days),
                "split_factor": 1.0,
                "div_cash": 0.0,
            },
            index=dates,
        )
    return panel


def test_run_strategy_is_deterministic_given_same_spec_and_panel():
    panel = _make_panel()
    spec = StrategySpec(factor="momentum", n_deciles=4)
    result_a = run_strategy(spec, panel)
    result_b = run_strategy(spec, panel)
    pd.testing.assert_series_equal(result_a.nav_history, result_b.nav_history)
    assert result_a.sharpe == result_b.sharpe


def test_api_call_and_dashboard_call_produce_identical_numbers():
    """The dashboard and the CLI both call run_strategy() directly — this
    test simulates both call sites with an identical spec and panel and
    asserts the results are equal, satisfying the M6 accept criteria."""
    panel = _make_panel(seed=1)
    spec_dict = {
        "factor": "low_vol",
        "construction": "long_only_top_decile",
        "n_deciles": 5,
    }

    cli_spec = StrategySpec(**spec_dict)
    dashboard_spec = StrategySpec.model_validate(spec_dict)

    cli_result = run_strategy(cli_spec, panel)
    dashboard_result = run_strategy(dashboard_spec, panel)

    pd.testing.assert_series_equal(cli_result.nav_history, dashboard_result.nav_history)
    assert cli_result.sharpe == dashboard_result.sharpe
    assert cli_result.turnover_total == dashboard_result.turnover_total


def test_different_factors_produce_different_results():
    panel = _make_panel(seed=2)
    momentum_result = run_strategy(StrategySpec(factor="momentum", n_deciles=4), panel)
    reversal_result = run_strategy(StrategySpec(factor="reversal", n_deciles=4), panel)
    assert momentum_result.sharpe != reversal_result.sharpe


def test_data_quality_filter_can_be_disabled():
    panel = _make_panel(seed=3)
    spec_filtered = StrategySpec(factor="momentum", n_deciles=4, apply_data_quality_filter=True)
    spec_unfiltered = StrategySpec(factor="momentum", n_deciles=4, apply_data_quality_filter=False)
    result_filtered = run_strategy(spec_filtered, panel)
    result_unfiltered = run_strategy(spec_unfiltered, panel)
    assert result_unfiltered.n_tickers >= result_filtered.n_tickers


def test_strategy_spec_is_plain_data_serializable():
    spec = StrategySpec(factor="momentum")
    as_dict = spec.model_dump()
    restored = StrategySpec.model_validate(as_dict)
    assert restored == spec
