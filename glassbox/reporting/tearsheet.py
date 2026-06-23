"""Static HTML tearsheet export for a StrategyResult.

Renders the same numbers the dashboard shows — equity curve, drawdown,
rolling Sharpe, and the deflated/haircut Sharpe next to the naive one — to
a self-contained HTML file with embedded matplotlib PNGs, so a tearsheet
can be shared without running Streamlit.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from glassbox.strategy import StrategyResult
from glassbox.validation.metrics import (
    deflated_sharpe_ratio,
    harvey_liu_haircut_sharpe,
    sharpe_ratio,
)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _equity_curve_png(navs: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(navs.index, navs.values)
    ax.set_title("Equity Curve")
    ax.set_ylabel("NAV ($)")
    return _fig_to_base64(fig)


def _drawdown_png(navs: pd.Series) -> str:
    running_max = navs.cummax()
    drawdown = (navs - running_max) / running_max
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.6)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    return _fig_to_base64(fig)


def _rolling_sharpe_png(returns: pd.Series, window: int = 63) -> str:
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * (252**0.5)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Rolling {window}-day Sharpe")
    return _fig_to_base64(fig)


def render_tearsheet_html(result: StrategyResult, n_trials: int = 3) -> str:
    navs = result.nav_history
    returns_series = navs.pct_change().dropna()

    # DSR/haircut need Sharpe and n_obs at the same frequency — annualized
    # Sharpe with daily n_obs pins every factor's DSR near 1.0 regardless
    # of quality (see glassbox.validation.metrics docstring).
    daily_sharpe = sharpe_ratio(result.returns, periods_per_year=1)
    dsr = deflated_sharpe_ratio(daily_sharpe, n_trials=n_trials, n_obs=len(result.returns))
    haircut_daily = harvey_liu_haircut_sharpe(
        daily_sharpe, n_trials=n_trials, n_obs=len(result.returns)
    )
    haircut = haircut_daily * (252**0.5)

    equity_b64 = _equity_curve_png(navs)
    drawdown_b64 = _drawdown_png(navs)
    rolling_b64 = _rolling_sharpe_png(returns_series)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>GLASSBOX Tearsheet: {result.spec.factor}</title></head>
<body style="font-family: sans-serif; max-width: 900px; margin: 2rem auto;">
  <h1>GLASSBOX Tearsheet — {result.spec.factor} ({result.spec.construction})</h1>
  <table style="border-collapse: collapse; width: 100%;">
    <tr><td><b>Naive Sharpe</b></td><td>{result.sharpe:.3f}</td></tr>
    <tr><td><b>Deflated Sharpe Ratio (n_trials={n_trials})</b></td><td>{dsr:.3f}</td></tr>
    <tr><td><b>Haircut Sharpe (n_trials={n_trials})</b></td><td>{haircut:.3f}</td></tr>
    <tr><td><b>Total cost paid</b></td><td>${result.cost_paid_total:,.2f}</td></tr>
    <tr><td><b>Total traded value (turnover)</b></td><td>${result.turnover_total:,.2f}</td></tr>
    <tr><td><b>Tickers in universe</b></td><td>{result.n_tickers}</td></tr>
    <tr><td><b>Rebalances</b></td><td>{result.n_rebalances}</td></tr>
    <tr><td><b>Excluded (data-quality flagged)</b></td><td>{len(result.excluded_tickers)}</td></tr>
  </table>
  <h2>Equity Curve</h2>
  <img src="data:image/png;base64,{equity_b64}" style="width: 100%;">
  <h2>Drawdown</h2>
  <img src="data:image/png;base64,{drawdown_b64}" style="width: 100%;">
  <h2>Rolling Sharpe</h2>
  <img src="data:image/png;base64,{rolling_b64}" style="width: 100%;">
</body>
</html>"""


def export_tearsheet(result: StrategyResult, out_path: Path, n_trials: int = 3) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_tearsheet_html(result, n_trials=n_trials)
    out_path.write_text(html)
    return out_path
