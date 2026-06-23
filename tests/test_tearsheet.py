"""M6 acceptance test: tearsheet export produces a self-contained HTML file
with the headline numbers (naive vs deflated vs haircut Sharpe) present."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glassbox.reporting.tearsheet import export_tearsheet, render_tearsheet_html
from glassbox.strategy import StrategyResult, StrategySpec


def _fake_result():
    dates = pd.bdate_range("2020-01-01", periods=120)
    navs = pd.Series(100_000.0 * (1.0003 ** np.arange(len(dates))), index=dates, name="nav")
    returns = navs.pct_change().dropna().to_numpy()
    return StrategyResult(
        spec=StrategySpec(factor="momentum"),
        nav_history=navs,
        returns=returns,
        sharpe=0.8,
        turnover_total=50_000.0,
        cost_paid_total=120.0,
        n_rebalances=6,
        n_tickers=42,
        excluded_tickers=["BADTICK"],
    )


def test_render_tearsheet_html_contains_headline_numbers():
    html = render_tearsheet_html(_fake_result(), n_trials=3)
    assert "Naive Sharpe" in html
    assert "Deflated Sharpe Ratio" in html
    assert "Haircut Sharpe" in html
    assert "momentum" in html
    assert "<img" in html


def test_export_tearsheet_writes_file(tmp_path):
    out_path = tmp_path / "tearsheet.html"
    written = export_tearsheet(_fake_result(), out_path)
    assert written.exists()
    assert written.read_text().startswith("<!DOCTYPE html>")
