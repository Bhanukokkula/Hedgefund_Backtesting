"""Streamlit dashboard: pick factor/universe/costs, see the honest numbers.

Calls glassbox.strategy.run_strategy() directly — the same function the
CLI (glassbox.validation.run_m5) and tests call — so a strategy defined
here and one defined via StrategySpec in code produce identical numbers by
construction, not by careful manual syncing.

Run as: .venv/bin/streamlit run glassbox/reporting/dashboard.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from glassbox.data.universe import load_price_panel
from glassbox.factors.ranking import decile_mean_scores
from glassbox.factors.scoring import low_vol_score, momentum_score, reversal_score
from glassbox.reporting.tearsheet import export_tearsheet
from glassbox.settings import settings
from glassbox.strategy import StrategySpec, run_strategy
from glassbox.validation.metrics import (
    deflated_sharpe_ratio,
    harvey_liu_haircut_sharpe,
    sharpe_ratio,
)

N_TRIALS = 3  # momentum, reversal, low_vol — the headline factors actually tried


@st.cache_data
def _load_panel():
    prices_dir = settings.parquet_dir / "prices"
    tickers = [p.stem for p in prices_dir.glob("*.parquet")]
    panel = load_price_panel(prices_dir, tickers)
    return {t: df for t, df in panel.items() if len(df) >= 252}


def main():
    st.set_page_config(page_title="GLASSBOX", layout="wide")
    st.title("GLASSBOX — Factor Backtest Dashboard")
    st.caption(
        "Every number here comes from the same run_strategy() function the CLI and "
        "tests call. The deflated and haircut Sharpe sit next to the naive one on "
        "purpose — that gap is the point of this project."
    )

    panel = _load_panel()
    if not panel:
        st.error(
            "No cached price data found under data/parquet/prices. Run the ingestion script first."
        )
        return

    with st.sidebar:
        st.header("Strategy")
        factor = st.selectbox("Factor", ["momentum", "reversal", "low_vol"])
        construction = st.selectbox("Construction", ["long_only_top_decile", "decile_long_short"])
        n_deciles = st.slider("Number of deciles", 2, 10, 10)
        commission_bps = st.number_input("Commission (bps)", value=settings.costs.commission_bps)
        half_spread_bps = st.number_input("Half-spread (bps)", value=settings.costs.half_spread_bps)
        apply_filter = st.checkbox("Apply data-quality filter", value=True)

    spec = StrategySpec(
        factor=factor,
        construction=construction,
        n_deciles=n_deciles,
        commission_bps=commission_bps,
        half_spread_bps=half_spread_bps,
        apply_data_quality_filter=apply_filter,
    )

    with st.spinner("Running backtest..."):
        result = run_strategy(spec, panel)

    naive = result.sharpe
    # DSR/haircut need Sharpe and n_obs at the same frequency — annualized
    # Sharpe with daily n_obs pins every factor's DSR near 1.0 regardless
    # of quality (see glassbox.validation.metrics docstring).
    daily_returns = result.returns
    daily_sharpe = sharpe_ratio(daily_returns, periods_per_year=1)
    dsr = deflated_sharpe_ratio(daily_sharpe, n_trials=N_TRIALS, n_obs=len(daily_returns))
    haircut_daily = harvey_liu_haircut_sharpe(
        daily_sharpe, n_trials=N_TRIALS, n_obs=len(daily_returns)
    )
    haircut = haircut_daily * (252**0.5)

    col1, col2, col3 = st.columns(3)
    col1.metric("Naive Sharpe", f"{naive:.3f}")
    col2.metric(
        "Deflated Sharpe Ratio", f"{dsr:.3f}", help="P(true Sharpe > 0), corrected for 3 trials"
    )
    col3.metric(
        "Haircut Sharpe", f"{haircut:.3f}", help="Bonferroni-style simplified Harvey-Liu haircut"
    )

    st.subheader("Equity Curve")
    fig = go.Figure(go.Scatter(x=result.nav_history.index, y=result.nav_history.values))
    fig.update_layout(yaxis_title="NAV ($)", height=350)
    st.plotly_chart(fig, use_container_width=True)

    navs = result.nav_history
    drawdown = (navs - navs.cummax()) / navs.cummax()
    st.subheader("Drawdown")
    dd_fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy"))
    dd_fig.update_layout(yaxis_title="Drawdown", height=250)
    st.plotly_chart(dd_fig, use_container_width=True)

    returns_series = navs.pct_change().dropna()
    rolling_sharpe = (returns_series.rolling(63).mean() / returns_series.rolling(63).std()) * (
        252**0.5
    )
    st.subheader("Rolling 63-day Sharpe")
    rs_fig = go.Figure(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values))
    rs_fig.update_layout(height=250)
    st.plotly_chart(rs_fig, use_container_width=True)

    st.subheader("Decile Monotonicity (most recent rebalance)")
    st.caption(
        "Mean factor score per decile across the full universe at the latest rebalance date."
    )
    try:
        from glassbox.engine.asof_accessor import ConcreteAsOfAccessor

        as_of = max(panel[next(iter(panel))].index)
        accessor = ConcreteAsOfAccessor(as_of_date=as_of.date(), panel=panel)
        if factor == "momentum":
            scores = momentum_score(
                accessor,
                list(panel),
                settings.factors.momentum.lookback_months,
                settings.factors.momentum.skip_months,
            )
        elif factor == "reversal":
            scores = reversal_score(
                accessor, list(panel), settings.factors.reversal.lookback_months
            )
        else:
            scores = low_vol_score(accessor, list(panel), settings.factors.low_vol.lookback_days)
        decile_means = decile_mean_scores(scores, n_deciles=n_deciles)
        bar_fig = go.Figure(go.Bar(x=list(decile_means.keys()), y=list(decile_means.values())))
        bar_fig.update_layout(xaxis_title="Decile", yaxis_title="Mean score", height=300)
        st.plotly_chart(bar_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not compute decile monotonicity chart: {exc}")

    st.subheader("Turnover & Costs")
    st.write(f"Total traded value: ${result.turnover_total:,.2f}")
    st.write(f"Total cost paid: ${result.cost_paid_total:,.2f}")
    st.write(f"Tickers excluded by data-quality filter: {len(result.excluded_tickers)}")

    if st.button("Export tearsheet HTML"):
        out_path = settings.cache_dir / f"tearsheet_{factor}.html"
        export_tearsheet(result, out_path, n_trials=N_TRIALS)
        st.success(f"Tearsheet written to {out_path}")


if __name__ == "__main__":
    main()
