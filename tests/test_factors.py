"""M4 acceptance tests: each price-derived factor produces a sensible
decile spread on synthetic data, and the fundamental seam correctly refuses
non-PIT inputs (Size, Value, Quality)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from glassbox.data.fmp import FMPProvider
from glassbox.engine.asof_accessor import ConcreteAsOfAccessor
from glassbox.factors.fundamental import QualityFactor, SizeFactor, ValueFactor
from glassbox.factors.ranking import (
    decile_long_short_weights,
    decile_mean_scores,
    decile_rank,
    long_only_top_decile_weights,
)
from glassbox.factors.scoring import low_vol_score, momentum_score, reversal_score


def _flat_panel(tickers_to_trend, n_days=300):
    """tickers_to_trend: dict[ticker, daily_drift] -> builds a deterministic
    price path close[t] = 100 * (1+drift)^t (zero noise, so factor scores
    are exactly orderable and decile monotonicity is unambiguous)."""
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    panel = {}
    for ticker, drift in tickers_to_trend.items():
        close = 100.0 * (1.0 + drift) ** np.arange(n_days)
        panel[ticker] = pd.DataFrame(
            {
                "close": close,
                "open": close,
                "high": close,
                "low": close,
                "volume": 1_000_000,
                "split_factor": 1.0,
                "div_cash": 0.0,
            },
            index=dates,
        )
    return panel, dates


def test_momentum_decile_monotonic_in_trend_strength():
    drifts = {f"T{i}": d for i, d in enumerate([-0.003, -0.001, 0.0, 0.001, 0.003, 0.005])}
    panel, dates = _flat_panel(drifts, n_days=300)
    as_of = dates[-1].date()
    accessor = ConcreteAsOfAccessor(as_of_date=as_of, panel=panel)
    scores = momentum_score(accessor, list(drifts), lookback_months=12, skip_months=1)

    ranked_by_drift = sorted(drifts, key=lambda t: drifts[t])
    ranked_by_score = sorted(scores, key=lambda t: scores[t])
    assert ranked_by_score == ranked_by_drift  # higher trend -> higher momentum score


def test_reversal_score_favors_recent_losers():
    drifts = {"WINNER": 0.01, "LOSER": -0.01, "FLAT": 0.0}
    panel, dates = _flat_panel(drifts, n_days=60)
    as_of = dates[-1].date()
    accessor = ConcreteAsOfAccessor(as_of_date=as_of, panel=panel)
    scores = reversal_score(accessor, list(drifts), lookback_months=1)
    assert scores["LOSER"] > scores["FLAT"] > scores["WINNER"]


def test_low_vol_score_favors_calm_names():
    dates = pd.bdate_range("2019-01-01", periods=200)
    rng = np.random.default_rng(0)
    calm = 100.0 + np.cumsum(rng.normal(0, 0.05, len(dates)))
    volatile = 100.0 + np.cumsum(rng.normal(0, 2.0, len(dates)))
    panel = {
        "CALM": pd.DataFrame(
            {
                "close": calm,
                "open": calm,
                "high": calm,
                "low": calm,
                "volume": 1,
                "split_factor": 1.0,
                "div_cash": 0.0,
            },
            index=dates,
        ),
        "VOLATILE": pd.DataFrame(
            {
                "close": volatile,
                "open": volatile,
                "high": volatile,
                "low": volatile,
                "volume": 1,
                "split_factor": 1.0,
                "div_cash": 0.0,
            },
            index=dates,
        ),
    }
    accessor = ConcreteAsOfAccessor(as_of_date=dates[-1].date(), panel=panel)
    scores = low_vol_score(accessor, ["CALM", "VOLATILE"], lookback_days=126)
    assert scores["CALM"] > scores["VOLATILE"]


def test_decile_long_short_weights_are_dollar_neutral():
    scores = {f"T{i}": float(i) for i in range(10)}
    weights = decile_long_short_weights(scores, n_deciles=10)
    assert abs(sum(weights.values())) < 1e-9
    assert weights["T9"] > 0  # highest score is long
    assert weights["T0"] < 0  # lowest score is short


def test_long_only_top_decile_weights_sum_to_one():
    scores = {f"T{i}": float(i) for i in range(10)}
    weights = long_only_top_decile_weights(scores, n_deciles=10)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w > 0 for w in weights.values())


def test_decile_mean_scores_monotonic():
    scores = {f"T{i}": float(i) for i in range(20)}
    means = decile_mean_scores(scores, n_deciles=10)
    ordered = [means[d] for d in sorted(means)]
    assert ordered == sorted(ordered)


def test_decile_rank_handles_small_samples_without_raising():
    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    ranks = decile_rank(scores, n_deciles=10)
    assert len(ranks) == 3


def test_decile_rank_drops_nan_and_inf_scores_without_raising():
    """Regression test: a single NaN/inf score (e.g. from a degenerate
    volatility calc on sparse real data) must not crash the whole ranking
    with an IntCastingNaNError, nor silently corrupt every other ticker's
    decile — it should simply be excluded."""
    scores = {"A": 1.0, "B": float("nan"), "C": 3.0, "D": float("inf")}
    ranks = decile_rank(scores, n_deciles=10)
    assert set(ranks) == {"A", "C"}


def test_size_factor_refuses_non_pit_fmp_provider():
    provider = FMPProvider(api_key="dummy")
    factor = SizeFactor(provider)
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 1, 1), panel={})
    result = factor.compute(accessor, ["AAPL"], date(2020, 1, 1))
    assert result.status == "refused_non_pit"
    assert result.scores is None
    assert "point-in-time" in result.reason


def test_value_and_quality_factors_also_refuse():
    accessor = ConcreteAsOfAccessor(as_of_date=date(2020, 1, 1), panel={})
    for factor in (ValueFactor(), QualityFactor()):
        result = factor.compute(accessor, ["AAPL"], date(2020, 1, 1))
        assert result.status == "refused_non_pit"
