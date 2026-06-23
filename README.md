# GLASSBOX

A glass-box (not black-box) cross-sectional factor backtesting platform,
built to resist the ways backtests lie. Result first, then method.

## The honest result

On a real, free-tier-constrained sample (117 real tickers cached; ~95 pass
the per-rebalance liquidity and data-quality filter, 1979–2026,
momentum/reversal/low-vol factors, monthly rebalanced, long-only top
decile):

| Factor | Net Sharpe | Gross Sharpe | Cost drag (bps) | Survivorship inflation (bps) | In-sample → out-of-sample Sharpe | Deflated Sharpe | Haircut Sharpe |
|---|---|---|---|---|---|---|---|
| Momentum | 0.35 | 0.39 | 189 | 253 | 0.25 → 0.71 | 0.94 | 0.29 |
| Reversal | 0.31 | 0.46 | 724 | 60 | 0.29 → 0.41 | 0.89 | 0.24 |
| Low-vol | 0.04 | 0.06 | 66 | 386 | 0.05 → 0.01 | 0.28 | 0.00 |

**Reversal survives costs at a 0.31 net Sharpe but loses 724 bps/year to
transaction costs alone — the most cost-sensitive of the three. Low-vol
barely clears zero net of costs (0.04 Sharpe), its out-of-sample Sharpe
(0.01) is close to noise, and the multiple-testing correction agrees:
Deflated Sharpe Ratio of 0.28 (vs. 0.94/0.89 for momentum/reversal) says
there's only a 28% chance this factor's true Sharpe is even positive once
you account for having tried 3 factors, and the haircut Sharpe collapses
it to exactly 0.0. Momentum and reversal, by contrast, both clear ~0.9 DSR
— genuinely hard to explain away as one lucky draw out of three. Low-vol
is also the single biggest survivorship-bias victim: a survivors-only
universe would have inflated its annualized return by 386 bps, more than
its entire net-of-cost edge.**

One result needs a second look rather than a celebration: momentum's
out-of-sample Sharpe (0.71) is roughly 3x its in-sample Sharpe (0.25) on
the single chronological 70/30 split this project runs. OOS beating IS by
that much on a ~95-name sample is much more likely a sign that the test
window happened to land in a strong-momentum regime than evidence of
robustness — a single split can't distinguish the two. Multiple
walk-forward folds (reporting the distribution of OOS Sharpe, not one
point estimate) would be the right next step before trusting that number;
treat it as a flagged uncertainty, not a confirmed result.

At the universe level (M1 gate, same 117 real tickers): **90.8%** of
universe-months in the survivorship-aware universe include a name that
later delisted, and a survivors-only universe shows **+72 bps** of
annualized return inflation versus the full universe — confirmed, real
survivorship bias, not a theoretical concern.

> **Honest caveats:**
> - **Sample size.** 117 cached tickers (~95 passing filters), not the
>   500-name target in `config.yaml`. Bounded by Tiingo's free-tier hourly
>   request allocation (~50 req/hour), not by design. The qualitative
>   findings — survivorship inflates results, costs erode the weaker
>   factors, low-vol doesn't survive deflation — would plausibly hold at
>   full scale, but magnitudes should be read as directional.
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
directly: 90.8% of universe-months touch an eventually-delisted name, and
a survivors-only universe's return is measurably inflated (+72 bps at the
universe level; up to +386 bps for the low-vol factor specifically).

**3. Transaction costs.** Every backtest reports gross and net Sharpe side
by side (`glassbox/validation/sensitivity.py`), with a cost model
(commission + half-spread + participation-rate-scaled market impact,
`glassbox/engine/costs.py`) applied at the moment a fill executes — always
the *next* bar's open after a signal, never the signal bar's own close
(`tests/test_backtest.py::test_no_same_bar_fill`). Reversal's 724 bps/year
cost drag is the headline honesty signal here: a factor that looks
attractive gross-of-cost and far less so net-of-cost.

**4. Multiple testing.** Every reported Sharpe sits next to a Deflated
Sharpe Ratio (Bailey & López de Prado) and a simplified Bonferroni-style
haircut Sharpe (`glassbox/validation/metrics.py`), tracking that 3 factor
configurations were actually tried. Both degrade monotonically as more
trials are simulated (tested in `tests/test_metrics.py`), and on the real
results table above the haircut doesn't just nudge the weakest factor — it
sends low-vol's Sharpe to exactly 0.0, while momentum and reversal hold up
at ~0.9 DSR. (This caught a real implementation bug during development:
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

- **Universe scale.** 117 cached tickers, not 500. A paid Tiingo tier (or
  patience across many hourly windows) would close this gap without any
  code changes — `glassbox/data/ingest.py` is already resumable and
  idempotent.
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
tests/         83 tests: adversarial as-of tests, hand-checked corporate
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
