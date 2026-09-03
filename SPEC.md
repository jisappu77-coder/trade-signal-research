# SPEC.md — `cryptolab`

**An evidence-gated research and backtesting platform for intraday cryptocurrency perpetual futures.**

> This document is a build specification for an AI coding agent. It is derived from a citation audit and
> edge-validation review of an August 2026 research paper proposing a 15-hypothesis intraday crypto framework.
> The review concluded that **most of the paper's proposed edges do not survive realistic transaction costs for a
> retail operator**. This spec deliberately encodes that conclusion: it builds the *validation machinery first*
> and permits only three signal families in v1.

---

## 0. Read this first

Three rules override everything else in this document:

1. **This is not a trading bot.** v1 has no order placement, no exchange keys with trade permission, no live
   execution. It is a research instrument whose output is a *verdict* on whether an edge exists, not a P&L.
2. **The cost model is not a parameter to tune.** It is a fixed adversary. Every result is reported net of
   costs or it is not reported.
3. **Every backtest is a statistical trial and must be counted.** The platform maintains a trial registry.
   Sharpe ratios are deflated by the number of trials run. An agent that runs 400 parameter sweeps and reports
   the best one has produced nothing.

---

## 1. Purpose and non-goals

### Purpose
Determine, for BTC/USDT and ETH/USDT perpetual futures, whether a small set of evidence-supported signals
produces a repeatable edge that survives realistic fees, slippage, funding, and multiple-testing correction.

### Explicit non-goals for v1
- Live or paper order execution
- Sub-minute / microstructure strategies
- Machine learning of any kind
- Multi-exchange arbitrage
- Portfolio construction across >2 assets
- Any UI beyond static HTML reports

### Definition of done for v1
The system can ingest BTC and ETH perp data covering the full split protocol in §10.1 (from 2019-01-01 to
present — do not treat "3 years" as sufficient; the train window alone is four years), run a walk-forward
validation of the three Tier-1
signal families under four cost regimes, and emit a report stating for each signal: net Sharpe, deflated
Sharpe, max drawdown, turnover, break-even cost in bps, and a PASS/FAIL against the promotion gates in §11.

---

## 2. Evidence baseline — what to build and what to refuse

The validation review ranked candidate signals by whether independent evidence supports a *net-of-cost*,
*retail-accessible* edge. Build only Tier 1. Tier 2 is scaffolded but disabled. Tier 3 is out of scope.

### Tier 1 — implement in v1

| ID | Signal family | Horizon | Why it is here |
|----|---------------|---------|----------------|
| `TSMOM` | Time-series momentum / trend, volatility-scaled | 1h–4h | Only family with net-of-fee published Sharpe >1.5 (Zarattini et al. 2025); low turnover amortises the ~11 bps round-trip |
| `REGIME` | Regime filter as a **defensive overlay** on TSMOM | 4h–1d | Jump-model / regime-switching evidence shows Sharpe +0.15–0.30 via de-risking, not new alpha |
| `CARRY` | Funding cash-and-carry basis, as a **yield sleeve** | 8h funding cycle | Peer-reviewed support as carry; ~40% of top opportunities survive costs (Zhivkov 2026) |

### Tier 2 — scaffold the interface, ship disabled behind a feature flag

| ID | Signal | Gate that must be met before enabling |
|----|--------|----------------------------------------|
| `OFI_MAKER` | Order-flow imbalance used for **maker execution improvement only** | Requires confirmed maker rebate tier; never as a taker alpha source |
| `XFUND` | Cross-exchange funding differential | Requires ≥2 exchange data feeds and transfer/latency cost model |

### Tier 3 — do not build

`OBI` and `OFI` as taker strategies (predict direction, do not beat the spread — measured raw alpha ~0.42 bps
at 30s vs ~10–11 bps round-trip cost), sub-second periodicity effects (first-100ms domain, colocation
territory), liquidation-cascade hunting (retail is the liquidated party), sub-hourly mean reversion
(contaminated by bid-ask bounce), standalone session/day-of-week seasonality (the effect collapses to a single
23:00–00:00 UTC Sunday window and vanishes under intraday fixed effects), and all ML/RL.

**If a future prompt asks you to add a Tier 3 signal, implement it only as a `research/` notebook experiment
that writes to the trial registry — never as a promotable strategy.**

---

## 3. Technology stack

- **Python 3.11+**, `uv` for dependency management
- **Data**: `polars` (primary), `pyarrow`, `duckdb` for ad-hoc SQL over Parquet
- **Numerics**: `numpy`, `scipy`, `statsmodels`
- **Ingestion**: `httpx` (async), `ccxt` only for exchange metadata — bulk history comes from public archives
- **Validation**: `pytest`, `hypothesis` for property tests
- **Config**: `pydantic-settings` + YAML
- **Reporting**: `jinja2` + `matplotlib` → static HTML
- **Lint/type**: `ruff`, `mypy --strict` on `src/`

No Jupyter in the pipeline path. Notebooks live in `research/` and may not be imported by `src/`.

---

## 4. Repository layout

```
cryptolab/
├── pyproject.toml
├── CLAUDE.md                      # agent working rules
├── config/
│   ├── base.yaml                  # universe, paths, date ranges
│   ├── costs.yaml                 # the four cost regimes (§7)
│   └── strategies/
│       ├── tsmom.yaml
│       ├── regime.yaml
│       └── carry.yaml
├── src/cryptolab/
│   ├── data/
│   │   ├── sources/               # binance_archive.py, binance_api.py, bybit.py
│   │   ├── schemas.py             # polars schemas, single source of truth
│   │   ├── ingest.py
│   │   ├── quality.py             # §6 data-quality gate
│   │   └── store.py               # Parquet partitioned read/write
│   ├── features/
│   │   ├── returns.py
│   │   ├── volatility.py
│   │   ├── derivatives.py         # funding, OI, basis
│   │   └── registry.py            # feature name → callable, with lookback declaration
│   ├── signals/
│   │   ├── base.py                # Signal ABC (§8)
│   │   ├── tsmom.py
│   │   ├── carry.py
│   │   └── regime.py
│   ├── backtest/
│   │   ├── engine.py              # event loop (§9)
│   │   ├── costs.py               # cost model (§7)
│   │   ├── portfolio.py
│   │   └── risk.py                # §13
│   ├── validation/
│   │   ├── walkforward.py
│   │   ├── deflated_sharpe.py
│   │   ├── pbo.py                 # probability of backtest overfitting
│   │   ├── registry.py            # trial registry (§10.4)
│   │   └── gates.py               # §11 promotion gates
│   ├── reporting/
│   └── cli.py                     # typer entrypoint
├── research/                      # notebooks; never imported by src/
├── tests/
└── data/                          # gitignored; parquet lake
```

---

## 5. Data layer

### 5.1 Sources and the gotchas that will bite

| Dataset | Source | Critical constraint |
|---------|--------|---------------------|
| OHLCV klines (1m, 5m, 1h) | `data.binance.vision` monthly ZIPs | Free, complete, use this — **not** the REST API for bulk |
| Funding rate history | Binance futures REST `/fapi/v1/fundingRate` | Paginate by `startTime`; 8h cadence; some symbols moved to 4h |
| **Open interest history** | Binance `/futures/data/openInterestHist` | **Only ~30 days of history is retrievable.** OI history cannot be backfilled. A daily collector must run from day one or any OI-derived hypothesis is permanently untestable on historical data. Flag this loudly at init. |
| Liquidations | Binance `!forceOrder@arr` websocket | Not backfillable either; and the public feed is throttled/partial — treat any liquidation series as lower-confidence |
| Mark / index price | Binance archive `markPriceKlines` | Needed to compute basis correctly; do **not** use last-traded price for basis |

**Symbol continuity**: contract specs change (tick size, multiplier, funding interval). Store a
`contract_spec_changes` table and refuse to backtest across an unhandled change without an explicit
`--allow-spec-change` flag.

### 5.2 Storage

Parquet, partitioned `data/{dataset}/exchange={ex}/symbol={sym}/year={y}/month={m}/`. All timestamps are
**UTC, milliseconds, int64**, bar-open convention. Every table carries `ingested_at` and `source_uri`.

### 5.3 Canonical schemas

```python
# ohlcv
open_time: i64, open: f64, high: f64, low: f64, close: f64,
volume: f64, quote_volume: f64, trades: i64,
taker_buy_base: f64, taker_buy_quote: f64, close_time: i64

# funding
funding_time: i64, symbol: str, funding_rate: f64, mark_price: f64

# open_interest
timestamp: i64, symbol: str, oi_base: f64, oi_quote: f64
```

---

## 6. Data quality gate

`quality.py` runs before any feature computation and **hard-fails** the pipeline on:

- Missing bars (gaps > 1 interval) — report count, location, and duration
- Duplicate timestamps
- Non-monotonic timestamps
- OHLC violations (`high < max(open, close)`, `low > min(open, close)`)
- Zero-volume runs > 10 bars on BTC/ETH (indicates an outage, not a quiet market)
- Funding rate outside [-0.75%, +0.75%] per interval (exchange cap breach ⇒ bad data)
- Price jumps > 20% in one 1m bar without a corresponding volume spike

Output a `DataQualityReport` per symbol-month, persisted. Backtests record the hash of the quality report of
every input they consumed. **A backtest over data that failed the gate is not a result.**

---

## 7. Cost model — the fixed adversary

`costs.py` implements four regimes. Every strategy is evaluated under all four; the **conservative** regime is
the headline number.

| Regime | Taker fee | Maker fee | Slippage model | Funding |
|--------|-----------|-----------|----------------|---------|
| Optimistic | 2.0 bps | 0.0 bps | half-spread | actual |
| Expected | 5.0 bps | 2.0 bps | half-spread + 1 bp | actual |
| **Conservative** | 5.5 bps | 2.0 bps | half-spread + 2 bps + impact | actual × 1.25 |
| Stressed | 7.5 bps | 3.0 bps | half-spread + 5 bps + 2× impact | actual × 2.0 |

Anchors: Binance and OKX regular tier ≈ 0.02% maker / 0.05% taker; Bybit ≈ 0.02% / 0.055%. A taker round-trip
is therefore **≈10–11 bps before slippage and funding**.

**Impact model**: `impact_bps = k * sqrt(order_notional / bar_quote_volume)`, `k` configurable, default 10.
Reject any fill where `order_notional > 0.01 * bar_quote_volume` and log it as a capacity breach.

**Funding**: applied at each settlement timestamp to open positions, sign-aware. Over a 5m–4h horizon funding
is usually second-order, but must be exact at extremes — a +0.51%/8h average implies ≈$153/day on a $10k long.

### 7.1 Mandatory derived metric: break-even cost

For every strategy the engine must report `breakeven_cost_bps` — the round-trip cost at which net Sharpe
crosses zero. This single number is more informative than the Sharpe itself. Surface it in every report header.

---

## 8. Signal interface

```python
class Signal(ABC):
    name: str
    tier: Literal[1, 2, 3]
    required_features: list[FeatureSpec]   # each declares its lookback in bars
    param_space: dict[str, ParamRange]     # declared up front; used for trial counting

    @abstractmethod
    def generate(self, features: pl.DataFrame, params: dict) -> pl.DataFrame:
        """Returns columns: timestamp, target_position (float in [-1, 1]), confidence.

        MUST be causal. Row i may only reference feature values at rows <= i.
        The engine enforces this with the shift test in §14.2.
        """
```

`target_position` is a **desired exposure**, not an order. The portfolio layer converts desire to orders and
applies the no-trade band (§9.3).

### 8.1 TSMOM specification

Primary: volatility-scaled time-series momentum on 1h bars, evaluated at 1h and 4h.

```
r_t         = log(close_t / close_{t-1})
sigma_t     = EWM stdev of r over halflife H, annualised          # H default 72 bars
signal_t    = sign( close_t / close_{t-L} - 1 )                   # L in {24, 48, 96, 168} bars
target_t    = signal_t * clip(sigma_target / sigma_t, 0, max_lev)
```

Defaults: `sigma_target = 0.40` annualised, `max_lev = 2.0`, `L = 96`, `H = 72`.

Required variants to test (this is the *entire* declared parameter space — do not expand it without
registering the expansion): `L ∈ {24, 48, 96, 168}`, `H ∈ {36, 72, 144}`, bar ∈ {1h, 4h}. That is 24 parameter
combinations. Register all 24 even if you only report one.

**Definition of `N`**: a trial is one `(signal, params, symbol, period)` tuple — **per symbol, not per
universe**. The 24 parameter combinations evaluated on `[BTCUSDT, ETHUSDT]` are therefore **48 trials**, and the
deflated Sharpe of *any one* of them is deflated by `N = 48`. Searching a second asset is a second search and
costs statistical power accordingly. `N` is always read from the registry (§10.4) at report time, never
hard-coded — the registry is the sole arbiter of this count.

Known caveat to encode as a test: momentum is concentrated in winners and **losers frequently rebound**
(Han, Kang & Ryu). Report long-leg and short-leg attribution separately. If the short leg has negative
expectancy net of costs, the report must say so rather than netting it out.

### 8.2 REGIME specification

A defensive overlay, not a standalone strategy. Two-state (calm / turbulent) classifier on 4h bars, using a
statistical jump model or, as the v1 baseline, a simpler rule:

```
turbulent_t = (realised_vol_t > percentile(realised_vol, 80, lookback=180d))
              OR (drawdown_from_60d_high < -0.20)
```

Effect: multiply TSMOM `target_position` by `regime_scalar` (default 0.5 in turbulent, 1.0 in calm).

**Acceptance criterion**: the overlay must reduce max drawdown by ≥15% relative to unfiltered TSMOM without
reducing net Sharpe by more than 0.10. If it fails this, report the failure — do not tune until it passes.

### 8.3 CARRY specification

Not directional. Delta-neutral funding capture: long spot / short perp when annualised funding exceeds a
threshold that clears costs.

```
funding_apr        = funding_rate * (8760 / funding_interval_hours)
entry_threshold    = 2 * round_trip_cost_apr_equivalent + margin_buffer
```

Must model: entry and exit costs on *both* legs, margin requirement and liquidation distance on the short
perp, funding-flip risk (position must exit when funding_apr falls below `exit_threshold`), and the empirical
prior that only ~40% of apparently attractive opportunities remain profitable after costs and spread
reversals. Report the realised hit rate against that 40% prior.

---

## 9. Backtest engine

### 9.1 Bar-close discipline
Signals are computed from bar `t` close. Orders are submitted at bar `t` close and **fill at bar `t+1` open**
with slippage. There is no intrabar fill logic in v1. If a future prompt asks for intrabar entries, refuse
unless tick or 1s data is present — inner-bar fills on 5m bars are the single most common source of inflated
crypto backtests.

### 9.2 Event loop
```
for each bar t:
    1. apply funding if t is a settlement timestamp
    2. mark existing positions to market
    3. check risk engine (§13) — may force flatten
    4. read target_position from signal (computed at t-1 close)
    5. diff against current position → order
    6. apply no-trade band
    7. fill at t open + slippage + fees
    8. record state
```

### 9.3 No-trade band
Do not rebalance unless `|target - current| > band` (default 0.10 of notional). This is the primary turnover
control and directly determines whether the strategy clears the cost hurdle.

### 9.4 Turnover accounting
Every run reports annualised turnover and `cost_drag_bps_per_year = turnover × round_trip_cost`. Print the
arithmetic explicitly in the report: at ~11 bps round-trip, a strategy trading 10× per day needs ~110 bps/day
of gross edge merely to break even.

---

## 10. Validation harness

### 10.1 Split protocol
```
train (in-sample)      : 2019-01-01 → 2022-12-31   — free exploration
validation             : 2023-01-01 → 2024-06-30   — limited, logged access
test (sealed)          : 2024-07-01 → present      — ONE touch, ever
```
The test set is enforced in code: `store.py` refuses to serve test-period data unless the caller passes a
one-time token issued by `validation/registry.py`, and the registry records the strategy hash that consumed
it. A second read of the test set by the same strategy family raises.

### 10.2 Walk-forward
Anchored and rolling variants. Train window ≥ 365d, test window 90d, step 90d. Report the distribution of
out-of-sample Sharpe across folds, not the aggregate. **Fold dispersion is the headline robustness metric** —
a strategy with mean OOS Sharpe 1.2 and fold stdev 1.4 is not a strategy.

### 10.3 Deflated Sharpe ratio
Implement per Bailey & López de Prado (2014).

```
SR0 = stdev(SR_trials) * [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]     γ = 0.5772157 (Euler–Mascheroni)

DSR = Φ( (SR_hat - SR0)·√(T-1) / √(1 - γ3·SR_hat + ((γ4-1)/4)·SR_hat²) )
```
where `N` = number of trials from the registry, `T` = number of return observations, `γ3`/`γ4` = skew and
kurtosis of the return series. Report DSR alongside every Sharpe. **A Sharpe reported without its DSR is a
bug.**

**Units**: `SR_hat`, `SR0` and `stdev(SR_trials)` must all be expressed at the **same (non-annualised) return
frequency** as the `T` observations. Annualising one side and not the other silently inflates DSR. The
implementation takes per-observation Sharpes only, and annualisation happens exclusively at the reporting
boundary.

### 10.4 Trial registry
An append-only SQLite table. Every `generate()` call with a distinct `(signal, params, symbol, period)`
tuple inserts a row before results are computed (per §8.1, trials are counted per symbol). `N` in the DSR formula reads from this table. Deleting rows
is a protocol violation; the table is hash-chained so tampering is detectable.

### 10.5 Probability of backtest overfitting
Implement combinatorially symmetric cross-validation (CSCV) → PBO. Report PBO for every strategy family.

---

## 11. Promotion gates

A strategy advances from `candidate` to `validated` only if **all** hold under the **conservative** cost
regime on out-of-sample data:

| Gate | Threshold |
|------|-----------|
| Net Sharpe (OOS) | > 1.0 |
| Deflated Sharpe | > 0.95 (i.e. p < 0.05 after trial correction) |
| PBO | < 0.30 |
| Walk-forward fold Sharpe stdev | < 0.75 × mean fold Sharpe |
| Max drawdown | < 35% |
| Break-even cost | > 2 × conservative round-trip cost |
| Parameter plateau | Sharpe within 25% of peak across ≥50% of neighbouring parameter grid |
| Beats controls | Outperforms BTC buy-and-hold, ETH buy-and-hold, and a random-entry control matched on exposure and turnover, on risk-adjusted net terms |

### 11.1 Kill criteria
Delete a candidate — do not tune it — if: net Sharpe < 0.3 under conservative costs; break-even cost < the
expected round-trip; or performance is concentrated in a single fold or a single quarter.

---

## 12. Reporting

One HTML report per strategy run, plus a comparison index. Required header block, in this order:

```
STRATEGY   TSMOM_L96_H72_1h              PERIOD  2024-07-01 → 2026-08-01  (OOS, sealed)
TRIALS N   48                            COSTS   conservative (5.5bps taker)
NET SHARPE 0.84    DEFLATED SHARPE 0.31  PBO     0.41
BREAKEVEN  6.2 bps round-trip            TURNOVER 41x/yr   COST DRAG 451 bps/yr
VERDICT    FAIL — deflated Sharpe below gate; edge does not clear cost hurdle
```

The verdict line is mandatory and must be computed, never written by hand. Reports for failed strategies are
retained and published in the index — the record of what did not work is the most valuable artifact this
system produces.

---

## 13. Risk engine

Independent of signal logic; can veto any target position. Limits: max gross leverage, max position per
symbol, daily loss limit, drawdown limit, max consecutive losses, data-staleness kill (no fresh bar within
2 intervals ⇒ flatten), and volatility circuit-breaker.

Implement a hard kill switch as a first-class object with an audit log, even though v1 places no orders — the
interface must be exercised in backtest so it is proven before any live wiring.

---

## 14. Testing requirements

### 14.1 Coverage
`src/cryptolab/backtest/` and `src/cryptolab/validation/` require ≥90% line coverage. Other modules ≥70%.

### 14.2 Mandatory anti-lookahead tests

1. **Shift test**: for every signal, recompute with all future data truncated at each timestamp; outputs must
   be bit-identical to the vectorised path. Any divergence is a lookahead bug.
2. **Shuffle test**: on randomly shuffled returns, every strategy must produce Sharpe indistinguishable from
   zero. A strategy that profits on shuffled data has lookahead.
3. **Cost monotonicity**: net Sharpe must be non-increasing as the cost regime worsens — asserted over the
   *fee and slippage* components only. Funding is **exempt**: the regimes scale funding magnitude (§7), and a
   funding-receiving position (any short perp, and all of CARRY) becomes *more* profitable as that multiplier
   rises. Asserting monotonicity over total P&L would fail correct code. The engine therefore decomposes P&L
   into `gross`, `fees`, `slippage` and `funding`, and the test asserts monotonicity of
   `gross - fees - slippage` while separately asserting that `fees + slippage` is non-decreasing.
4. **Zero-signal test**: a signal returning constant 0 must produce exactly zero P&L and zero cost.
5. **Funding sign test**: a long position through a positive funding settlement must lose exactly
   `notional × funding_rate`.

### 14.3 Property tests
Portfolio accounting invariants under `hypothesis`: cash + position value + realised costs = equity, always.

---

## 15. Build phases

| Phase | Deliverable | Acceptance |
|-------|-------------|-----------|
| 1 | Data layer: ingest, store, quality gate | 3 years BTC+ETH 1h/5m klines + funding ingested; quality report clean; OI collector running daily |
| 2 | Cost model + backtest engine | All §14.2 tests pass; zero-signal and shuffle tests green |
| 3 | Validation harness | DSR, PBO, walk-forward, trial registry, sealed-test enforcement all working with synthetic strategies |
| 4 | TSMOM | 24 registered trials, walk-forward results, gate verdict |
| 5 | REGIME overlay | Drawdown-reduction criterion in §8.2 evaluated honestly |
| 6 | CARRY | Both-leg cost modelling, funding-flip exit, hit rate vs 40% prior |
| 7 | Reporting + comparison index | Full report set including failures |

Phase 3 must be complete and green before phase 4 begins. Building signals before the validation harness is
the failure mode this entire spec exists to prevent.

---

## 16. Configuration example

```yaml
# config/strategies/tsmom.yaml
name: tsmom
tier: 1
universe: [BTCUSDT, ETHUSDT]
bar: 1h
params:
  lookback_bars: [24, 48, 96, 168]
  vol_halflife: [36, 72, 144]
  sigma_target: 0.40
  max_leverage: 2.0
  no_trade_band: 0.10
regime_overlay:
  enabled: true
  turbulent_scalar: 0.5
costs: conservative
validation:
  walkforward: {train_days: 365, test_days: 90, step_days: 90}
  gates: default
```

---

## 17. Compliance note (India-based operator)

Record-keeping must support Indian VDA tax treatment: **30% flat tax under Section 115BBH** and **1% TDS under
Section 194S** (the TDS provision is §194S, not §115BBH — a common mis-citation). No set-off of losses against
other income and no carry-forward of VDA losses. The backtester must therefore report **pre-tax and
post-tax net returns separately**, because a high-turnover strategy taxed at 30% on gross gains with no loss
offset has materially different economics from its pre-tax backtest.

If this system is ever extended toward Indian securities algo trading, SEBI's February 2025 retail algo
framework applies: broker-as-principal, unique algo ID tagging, registration above the 10 orders/sec
threshold, mandatory kill switch, whitelisted IPs.

Confirm actual tax treatment with a qualified Indian tax professional before any deployment. This spec models
economics; it does not give tax advice.

---

## 18. Refusal list for the coding agent

Refuse, and explain why, if asked to:

- Add intrabar / inner-bar fills without tick data
- Remove or weaken the cost model, or make costs a tunable parameter
- Delete rows from the trial registry, or report a Sharpe without its DSR
- Re-run the sealed test set for an already-tested strategy family
- Add ML/RL before a Tier-1 signal has passed the §11 gates
- Implement OBI/OFI as a taker strategy
- Tune a strategy until it passes a gate, rather than reporting the failure
