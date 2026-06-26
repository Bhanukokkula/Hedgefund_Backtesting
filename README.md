# GLASSBOX

A glass-box (not black-box) cross-sectional factor backtesting platform,
built to resist the ways backtests lie. Result first, then method.

## The honest result

On a real, free-tier-constrained sample (476 real tickers cached; 238 pass
the per-rebalance liquidity and data-quality filter, 1979–2026,
momentum/reversal/low-vol factors, monthly rebalanced, long-only top
decile):

| Factor | Net Sharpe | Gross Sharpe | Cost drag (bps) | Survivorship inflation (bps) | In-sample → out-of-sample Sharpe | Deflated Sharpe | Haircut Sharpe |
|---|---|---|---|---|---|---|---|
| Momentum | 0.35 | 0.41 | 219 | -26 | 0.32 → 0.48 | 0.96 | 0.30 |
| Reversal | 0.22 | 0.43 | 748 | 146 | 0.30 → 0.08 | 0.79 | 0.14 |
| Low-vol | **0.01** | 0.04 | 64 | **285** | 0.00 → 0.03 | **0.21** | **0.00** |

**Low-vol is, after two rounds of data-quality auditing, essentially flat
(0.01 net Sharpe — indistinguishable from zero), and the multiple-testing
correction agrees emphatically: a Deflated Sharpe Ratio of 0.21 says
there's roughly a 1-in-5 chance this factor's true Sharpe is even
positive once you account for trying 3 factors, and the haircut Sharpe
collapses it to exactly 0.0.** Survivorship bias would still have made it
look like it works: a survivors-only universe shows a respectable +0.16
Sharpe for the same factor (+285 bps of inflation) — read only that
number, as a naive backtest would, and you'd ship a factor with no real
edge believing it has one. Reversal is the cost-sensitivity story: it
survives at a 0.22 net Sharpe but loses 748 bps/year to transaction costs
alone (the most cost-sensitive of the three), and its out-of-sample Sharpe
(0.08) is a fraction of its in-sample Sharpe (0.30) — a real decay signal,
not noise, on this sample size. Momentum is the most robust of the three:
0.96 DSR, out-of-sample Sharpe holds up well (0.48 vs. 0.32 in-sample),
and costs take a real but survivable 219 bps/year bite. Interestingly,
momentum's survivorship "inflation" is slightly *negative* here (-26 bps)
— its full-universe Sharpe is marginally higher than survivors-only,
plausibly because momentum sometimes catches names right before a
buyout-driven delisting (a price spike, not a failure) — a reminder that
survivorship bias has a typical direction, not a guaranteed one.

**Two real, escalating data-quality bugs surfaced and got fixed producing
this table — worth narrating in full rather than just showing the final
numbers, since this is the project's whole thesis in miniature.** The
first version had low-vol at -0.13 Sharpe (looked like it loses money).
Auditing the factor (not retuning it) found its long ("safest") decile
dominated by **preferred shares** (`BAC-P-W`, `MS-P-E` — Tiingo's "-P-X"
ticker convention) — bond-like instruments, mechanically low-volatility
without the equity risk premium, never supposed to be in a common-equity
study. Excluding them (`glassbox/data/candidates.py::is_preferred_share`)
moved low-vol to -0.08 — still negative. A second pass found the remaining
contamination: **stale/thinly-traded names** with up to 65% identical-to
-previous-day closes, artificially flat without being genuinely low-risk
(`glassbox/validation/m1_report.py::detect_stale_pricing`). Excluding
those too moved low-vol to its final 0.01 — flat, not losing money, and
not a real edge either. Each fix changed the magnitude *and* moved the
result in different directions (worse, then better) — which is exactly
the point: the fixes were applied because they were correct, not because
of which way they pushed the number. See
`tests/test_candidates.py::test_preferred_shares_excluded` and
`tests/test_m1_report.py::test_detect_stale_pricing_flags_mostly_flat_tickers`.

One caveat on the walk-forward numbers: both factors' IS/OOS gap comes
from a single chronological 70/30 split. Reversal's apparent decay and
momentum's apparent stability are each consistent with genuine signal
quality, but neither can fully rule out which window happened to land in
which regime — multiple walk-forward folds (reporting a distribution, not
one point estimate) would be the rigorous next step.

At the universe level (M1 gate, a 270-ticker subsample: 134 active + 136
delisted): **99.4%** of universe-months in the survivorship-aware universe
include a name that later delisted, and a survivors-only universe shows
**+89 bps** of annualized return inflation versus the full universe —
confirmed, real survivorship bias, not a theoretical concern, and visible
again at the individual-factor level above (low-vol, +285 bps).

> **Honest caveats:**
> - **Sample size.** 476 cached tickers (238 passing filters for M5), not
>   the 500-name target in `config.yaml`. Bounded by a hard wall on
>   Tiingo's free tier — 500 **unique symbol lookups per calendar month**,
>   separate from and in addition to the ~50/hour rate limit — not by
>   design. The qualitative findings — survivorship inflates results,
>   costs erode the weaker factors, low-vol doesn't survive deflation —
>   would plausibly hold at full scale, but magnitudes should be read as
>   directional.
> - **No margin/leverage model, by choice, not by accident.** An earlier
>   dollar-neutral long-short construction (100% long / 100% short, 200%
>   gross) was tried first and is mathematically able to lever past its
>   own capital with no risk constraint — on this noisy small-cap sample
>   it pushed NAV negative. Rather than bolt on an ad hoc margin model,
>   the headline results use long-only top-decile, which is structurally
>   incapable of that failure mode (fully invested, no shorting, no
>   leverage) and is one of the two constructions M4 was built to support.
> - **Single-split walk-forward.** See the momentum note above — one
>   chronological split can flag overfitting but can't rule out regime
>   luck in either direction.

## How GLASSBOX resists each way backtests lie

**1. Look-ahead bias.** Every data read goes through an `AsOfAccessor`
bound to one monotonic clock (`glassbox/engine/asof_accessor.py`).
Corporate-action adjustment is reconstructed from raw close + split factor
+ dividend cash, truncated to the as-of date — never Tiingo's
fully-adjusted series, which bakes in future splits. The adversarial test
suite (`tests/test_asof_adversarial.py`) tries to read tomorrow's price and
tomorrow's split and asserts both are refused. Separately, a deliberate
look-ahead audit (`glassbox/validation/lookahead_audit.py`) injects a
20-trading-day peek into a synthetic strategy and shows it producing a
measurable, positive NAV improvement — proof the engine can both refuse
the leak when used correctly and quantify what the leak would have been
worth if it hadn't been.

**2. Survivorship bias.** The investable universe at each monthly
rebalance is built from trailing dollar volume *as known at that date*,
explicitly including names that later delisted
(`glassbox/data/universe.py`). The M1 validation gate measures this
directly: 99.4% of universe-months touch an eventually-delisted name, and
a survivors-only universe's return is measurably inflated (+89 bps at the
universe level; +285 bps for low-vol specifically — enough to make a
factor with no real edge look like it has one).

**3. Transaction costs.** Every backtest reports gross and net Sharpe side
by side (`glassbox/validation/sensitivity.py`), with a cost model
(commission + half-spread + participation-rate-scaled market impact,
`glassbox/engine/costs.py`) applied at the moment a fill executes — always
the *next* bar's open after a signal, never the signal bar's own close
(`tests/test_backtest.py::test_no_same_bar_fill`). Reversal's 748 bps/year
cost drag is the headline honesty signal here: a factor that looks
attractive gross-of-cost (0.43 Sharpe) and far less so net-of-cost (0.22).

**4. Multiple testing.** Every reported Sharpe sits next to a Deflated
Sharpe Ratio (Bailey & López de Prado) and a simplified Bonferroni-style
haircut Sharpe (`glassbox/validation/metrics.py`), tracking that 3 factor
configurations were actually tried. Both degrade monotonically as more
trials are simulated (tested in `tests/test_metrics.py`), and on the real
results table above the haircut doesn't just nudge the weakest factor — it
sends low-vol's Sharpe to exactly 0.0 (DSR 0.21 — roughly a 1-in-5 chance
this factor's true Sharpe is even positive), while momentum holds up at
0.96 DSR and reversal at 0.79. (This caught a real implementation bug during
development:
the Sharpe passed into the DSR/haircut formulas must be at the same
frequency as the observation count — annualized Sharpe paired with a
daily observation count inflates the test statistic and pins every
factor's DSR near 1.0 regardless of quality, which is exactly what
the first version of this table showed before the fix.
`tests/test_metrics.py::test_dsr_with_annualized_sharpe_and_daily_n_obs_saturates_uninformatively`
documents the failure mode so it can't silently reappear.)

## The core invariant

```
No computation may read any datum whose knowable-date is later than the
simulation's current as-of clock.
```

## Architecture

```
                    ┌─────────────────────────┐
                    │   Tiingo / FMP (network) │   <- only touched by
                    │   glassbox/data/ingest.py│      ingestion scripts
                    └────────────┬────────────┘
                                 │ parquet (cached, offline from here on)
                                 ▼
                    ┌─────────────────────────┐
                    │  LocalParquetProvider /  │
                    │  raw OHLCV panel          │
                    └────────────┬────────────┘
                                 ▼
        ┌────────────────────────────────────────────┐
        │   AsOfAccessor(as_of_date)                  │  <- the chokepoint:
        │   - price_series() truncates to as_of_date  │     every read goes
        │   - reconstructs adjustment from unadjusted │     through here
        │   - universe()/is_tradable() also truncated │
        └───────────────┬──────────────────────────────┘
                         │ (advance_to() moves the clock forward, never back)
                         ▼
        ┌────────────────────────────────────────────┐
        │   BacktestEngine event loop, per date:      │
        │   1. MarketOpen  — fill PREVIOUS day's order │
        │   2. Rebalance   — score today's close,      │
        │                    queue order for NEXT open │
        │   3. MarketClose — mark NAV                  │
        └───────────────┬──────────────────────────────┘
                         ▼
        ┌────────────────────────────────────────────┐
        │ glassbox.strategy.run_strategy(spec, panel) │  <- single declarative
        │ (called identically by CLI, tests, and the  │     entry point
        │  Streamlit dashboard — one implementation)  │
        └───────────────┬──────────────────────────────┘
                         ▼
        ┌────────────────────────────────────────────┐
        │ glassbox/validation/*  (the lie-resistance  │
        │ suite: cost & survivorship sensitivity,     │
        │ look-ahead audit, walk-forward, DSR/haircut)│
        └────────────────────────────────────────────┘
```

## Scope

Cross-sectional, price-derived equity factors: momentum (12-1), short-term
reversal (1-month), low-volatility. **Size is deliberately not shipped as
a headline factor** — market cap needs point-in-time shares outstanding,
and no free source provides that at the depth/scale this project needs
(Tiingo's fundamentals API is Dow-30-only on the free plan; FMP's
historical market cap only returns ~60 trading days; FMP's shares-float is
a current snapshot with no history). Size, Value, and Quality are all
defined through `glassbox/factors/fundamental.py`'s PIT-integrity seam,
which **refuses** to compute on non-point-in-time inputs rather than
silently faking them — see `tests/test_factors.py::test_size_factor_refuses_non_pit_fmp_provider`.

## Limitations / what I'd do next

- **Universe scale.** 476 cached tickers, not 500 — 24 short, blocked by
  Tiingo's hard monthly cap (500 unique-symbol lookups/calendar month,
  separate from the hourly rate limit) until next month's reset or a paid
  plan. `glassbox/data/ingest.py` is resumable and idempotent, so closing
  the gap needs zero code changes, just time.
- **Point-in-time fundamentals.** Would need Compustat/CRSP point-in-time
  data (not free) to ship Value/Quality/Size as real headline factors.
- **No margin/leverage model.** See the honest caveats above — this is a
  scoping decision (long-only top-decile, structurally NAV-safe), not a
  gap; a production version would need explicit leverage/margin accounting
  to support long-short construction safely.
- **Single-split walk-forward.** Multiple folds with a reported OOS
  distribution would replace the current single 70/30 point estimate —
  see momentum's IS/OOS note above for why this matters concretely.
- **No intraday execution, no options/derivatives, no non-US markets** —
  out of scope by design, not by oversight.
- **Regime analysis** (factor performance conditional on macro regime)
  would be a natural extension of the walk-forward harness already built.

## Layout

```
glassbox/
  data/        DataProvider protocol + Tiingo/FMP/LocalParquet implementations,
               candidate universe construction, survivorship-aware universe
  engine/      AsOfAccessor, adjustment reconstruction, event-driven backtest,
               cost model, portfolio accounting
  factors/     momentum, reversal, low-vol scoring + decile ranking;
               the fundamental PIT-refusal seam (Size/Value/Quality)
  strategy.py  the declarative StrategySpec -> run_strategy() API
  validation/  the lie-resistance suite: M1 gate, metrics (Sharpe/DSR/haircut),
               survivorship & cost sensitivity, look-ahead audit, walk-forward
  reporting/   Streamlit dashboard, HTML tearsheet export
tests/         89 tests: adversarial as-of tests, hand-checked corporate
               actions, hand-reconciled backtest P&L, synthetic-fixture
               factor/metric tests, declarative-API equivalence tests
results.json   every reported metric as a {target, result} pair
config.yaml    single source of truth for engine/data/cost parameters
```

## Setup

```
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # fill in TIINGO_API_KEY, FMP_API_KEY
.venv/bin/pytest

# pull real data (paced under Tiingo's free-tier hourly cap; resumable)
.venv/bin/python -m glassbox.data.ingest

# the M1 stop-and-report gate
.venv/bin/python -m glassbox.validation.m1_report

# the real per-factor M5 results table
.venv/bin/python -m glassbox.validation.run_m5

# the dashboard
.venv/bin/streamlit run glassbox/reporting/dashboard.py
```
