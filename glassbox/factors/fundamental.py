"""The fundamental-factor seam: an interface for value/quality/size factors
that DETECTS and REFUSES non-point-in-time inputs rather than silently
trusting them.

This project's scope is locked to price-derived factors (momentum,
reversal, low-vol) precisely because free fundamentals/shares-outstanding
data is not restatement-aware point-in-time. Size — conceptually
price-derived (market cap = price x shares outstanding) — still routes
through this seam: a real historical, point-in-time shares-outstanding
series at the scale this project needs (hundreds of names, decades) is not
available for free (see glassbox.data.fmp and glassbox.data.tiingo for what
was actually tried: Tiingo fundamentals is Dow-30-only on the free plan,
FMP's historical-market-cap free tier only returns ~60 trading days, and
FMP's shares-float is a current snapshot with no history at all).

The refusal itself — not a computed Size factor — is the headline result
for this seam. Value and quality are defined the same way but with no real
provider behind them at all; they exist to prove the interface
generalizes, not to ship a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from glassbox.engine.asof_accessor import ConcreteAsOfAccessor


class PITRefusalError(RuntimeError):
    pass


class FundamentalDataSource(Protocol):
    def is_point_in_time(self) -> bool: ...


@dataclass
class FundamentalFactorResult:
    name: str
    status: str  # "computed" | "refused_non_pit"
    reason: str | None = None
    scores: dict[str, float] | None = None


def require_point_in_time(source: FundamentalDataSource, factor_name: str) -> None:
    if not source.is_point_in_time():
        raise PITRefusalError(
            f"'{factor_name}' input is not point-in-time-safe: "
            f"{type(source).__name__}.is_point_in_time() returned False. "
            "Refusing to compute rather than silently using restated/current "
            "data for historical dates."
        )


class FundamentalFactor:
    """Base class: subclasses implement `_compute_scores`; `compute()`
    always checks PIT-safety first and converts a refusal into a
    FundamentalFactorResult instead of letting an exception escape into the
    backtest loop — the loop should be able to record "this factor refused
    itself" as data, not crash."""

    name: str = "fundamental"

    def __init__(self, data_source: FundamentalDataSource):
        self.data_source = data_source

    def compute(
        self, accessor: ConcreteAsOfAccessor, tickers: list[str], as_of_date: date
    ) -> FundamentalFactorResult:
        try:
            require_point_in_time(self.data_source, self.name)
        except PITRefusalError as exc:
            return FundamentalFactorResult(
                name=self.name, status="refused_non_pit", reason=str(exc)
            )
        scores = self._compute_scores(accessor, tickers, as_of_date)
        return FundamentalFactorResult(name=self.name, status="computed", scores=scores)

    def _compute_scores(
        self, accessor: ConcreteAsOfAccessor, tickers: list[str], as_of_date: date
    ) -> dict[str, float]:
        raise NotImplementedError


class SizeFactor(FundamentalFactor):
    """Market cap = latest price x shares outstanding. See module docstring
    for why this never actually reaches `_compute_scores` with this
    project's free data sources — `data_source` is an FMPProvider, whose
    `is_point_in_time()` returns False, so `compute()` always refuses."""

    name = "size"

    def _compute_scores(
        self, accessor: ConcreteAsOfAccessor, tickers: list[str], as_of_date: date
    ) -> dict[str, float]:
        scores = {}
        for ticker in tickers:
            shares = self.data_source.get_shares_outstanding(ticker, as_of_date)
            price = accessor.latest_price(ticker)
            if shares and price:
                scores[ticker] = price * shares
        return scores


class _NoProviderConfigured:
    """Stands in for value/quality factors, which have no fundamentals
    provider plugged in at all — confirming the interface generalizes to
    factors this project never claims to ship, not just Size."""

    def is_point_in_time(self) -> bool:
        return False

    def get_shares_outstanding(self, ticker: str, as_of: date) -> float | None:
        raise NotImplementedError


class ValueFactor(FundamentalFactor):
    name = "value"

    def __init__(self):
        super().__init__(_NoProviderConfigured())

    def _compute_scores(self, accessor, tickers, as_of_date):
        raise NotImplementedError("no fundamentals provider configured for value")


class QualityFactor(FundamentalFactor):
    name = "quality"

    def __init__(self):
        super().__init__(_NoProviderConfigured())

    def _compute_scores(self, accessor, tickers, as_of_date):
        raise NotImplementedError("no fundamentals provider configured for quality")
