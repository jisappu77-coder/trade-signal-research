# Real-data validation of the Phase 1–3 harness

**Date:** 2026-08-27  **Data:** Binance USD-M perpetuals, public sources only
**Period:** 2020-01-01 → 2024-06-30 (train + validation). **The sealed test period was never opened.**

This document records what the instrument did when pointed at real market data, and the four engine
defects that only real data exposed. It is not a strategy report — no Tier-1 signal exists yet.

---

## 1. Data sources actually used

| Dataset | Source | Result |
|---|---|---|
| 1h klines | `data.binance.vision` monthly ZIPs | **works** — 54 months × 2 symbols, no 404s |
| Funding rate | `data.binance.vision` monthly ZIPs | **works** — added in this change |
| Funding rate | `fapi.binance.com` REST | **HTTP 451** from this host |
| Open interest | `fapi.binance.com` REST | **HTTP 451** from this host |

`fapi.binance.com` returns `451 Service unavailable from a restricted location` for every endpoint,
including `/fapi/v1/time`. This is Binance geo-blocking the caller, not a proxy failure — the
archive host serves the same account fine.

**Consequence:** an archive-backed funding path was added (`ingest_funding_archive`, now the CLI
default). It is the better source anyway: no pagination, no rate limits, and it states
`funding_interval_hours` explicitly rather than leaving the cadence to be inferred from settlement
gaps — which §5.1 warns about, since some symbols moved from 8h to 4h.

**Open interest remains unavailable from here.** There is no archive fallback for it, so the §5.1
constraint is untouched: OI history is not backfillable, and the daily collector must run from a
host Binance will serve.

**Archive coverage starts 2020-01**, not 2019-01. The monthly UM-futures archive has no earlier
months for either symbol. The §10.1 train window opens 2019-01-01, so the first year of it cannot
be filled from this source.

## 2. Data quality (§6 gate)

| | BTCUSDT | ETHUSDT |
|---|---|---|
| Bars (1h) | 39,409 | 39,409 |
| Funding settlements | 4,927 | 4,927 |
| OHLCV gate | **PASS** — no findings | **PASS** — no findings |
| Funding gate | **PASS** | **PASS** |
| Funding interval | 8h throughout | 8h throughout |
| Price first → last | 7,171.55 → 61,016.30 | 128.82 → 3,384.93 |
| Mean funding | +15.06% APR | +18.29% APR |
| Funding range (per 8h) | −0.300% … +0.300% | −0.356% … +0.375% |

54 months of real data across both symbols passed the gate with zero findings: no gaps, no
duplicates, no OHLC violations, no zero-volume runs, no unsupported price jumps, no funding-cap
breaches. Funding never approached the ±0.75% cap the gate checks.

## 3. Capacity is the binding constraint, not the signal

The §7 participation limit (1% of bar quote volume) bites immediately on real data.

| | BTCUSDT | ETHUSDT |
|---|---|---|
| Thinnest 1h bar | $3.16M | **$0.62M** |
| 1st percentile bar | $22.9M | $2.82M |
| Max book that can turn a full 2× position in one bar | ~$15.8k | **~$3.1k** |

A $100k book at 2× leverage cannot rebalance in one bar on ETH's thinnest hours without breaching
the limit. This is a real result about tradeable size, and it is why the runs below use a $25k book.
The limit was not relaxed to make the runs fit.

## 4. Four defects real data exposed

Synthetic bars have constant, generous volume and well-behaved paths. Real bars do not, and four
bugs surfaced that 230 passing tests had not:

1. **`max_gross_leverage` was declared but never enforced.** The risk engine clamped the *target*
   and never checked *realised* exposure. Combined with (2), a book reached **32,000× leverage**
   and then traded on from **−$67M equity**. `RiskEngine.check` now takes `gross_notional` and
   trips on the realised figure.

2. **Capacity rejection trapped the book.** §7 rejects an over-limit fill. Applied to a
   *risk-reducing* order that means the book cannot delever — so leverage rises, the next
   corrective order is larger, and it too is refused. A death spiral built out of a safety limit.
   Risk-reducing orders are now worked down to the participation cap across bars; risk-*increasing*
   orders are still refused outright. The partial pays full impact, so this is not a discount.

3. **Insolvency did not stop the run.** Equity going through zero produced a P&L path that
   continued for another 16,858 bars. Runs now halt at insolvency and the result is flagged
   `went_bankrupt`.

4. **The no-trade band and the leverage limit were mutually incompatible.** §9.3 lets exposure
   drift 10% before rebalancing; the leverage check tripped at 5% drift. A strategy doing exactly
   what §9.3 prescribes was killed for it. `leverage_tolerance` now defaults to 0.15, and a test
   asserts it exceeds the default band.

A fifth, cosmetic: flattening left a ~1e-16 unit residue that accrued funding at every settlement
forever. Positions now snap to exact zero.

Also corrected in the same pass: `on_capacity_breach` now defaults to `"reject"` (reject the fill,
log the breach) which is what §7 actually specifies. The previous default aborted the whole run.

## 5. Harness behaviour on real data

All §14.2 checks re-run against real prices:

| Check | Result |
|---|---|
| Shift test (bit-identical under truncation) | **PASS** at 1k / 10k / 30k bars, both symbols |
| Shuffle test (gross Sharpe on shuffled returns) | **PASS** — −0.033 annualised, 8 seeds, within 1 s.e. of zero |
| Zero-signal | **PASS** — exactly 0.00 P&L, 0.00 costs |
| Cost monotonicity | **PASS** — net return non-increasing, cost/traded non-decreasing across all four regimes |
| Funding sign | **PASS** — 4,926 of 4,927 real settlements applied; long paid $206,100, short received $1,397 |
| Sealed period | **HOLDS** — a 2024-07 read is refused without a token |
| Trial registry | **N = 8**, hash chain intact |

Cost monotonicity on real BTC data, one probe configuration:

| Regime | Cost per unit traded | Net return | Net Sharpe (ann.) |
|---|---|---|---|
| optimistic | 3.00 bps | +263.8% | 0.87 |
| expected | 7.00 bps | −23.1% | −0.37 |
| conservative | 8.95 bps | −27.3% | −0.46 |
| stressed | 14.38 bps | −37.7% | −0.71 |

The gap between the first two rows is the whole thesis of this project in one line: a strategy that
looks like a 264% return at optimistic costs is a **loss** at merely expected ones.

## 6. What the gates said

The probe used here (`MomentumProbe`) is **Tier 3 and explicitly non-promotable** — it exists to
exercise the harness. It is not TSMOM, which belongs to Phase 4. Its numbers are reported only to
show the machinery producing a verdict.

```
STRATEGY   PROBE_L24_1h_ETHUSDT          PERIOD  2020-01-01 → 2024-06-30  (train+validation)
TRIALS N   8                             COSTS   conservative (5.5bps taker)
NET SHARPE -1.49   DEFLATED SHARPE 0.90  PBO     0.50
BREAKEVEN  175.4 bps round-trip          TURNOVER 95x/yr   COST DRAG 1427 bps/yr
VERDICT    KILLED — net Sharpe -1.49 < 0.3 under conservative costs (SPEC.md §11.1: do not tune)
```

**The instrument caught exactly what it is built to catch.** Full-sample, that configuration posts
an annualised net Sharpe of **+1.26** — a number that, reported alone, reads like an edge. Its
walk-forward out-of-sample mean across 14 folds is **−1.49**, with fold Sharpes ranging from +2.44
to −5.55. PBO is 0.50: selecting the best configuration in-sample is a coin flip out of sample.

Deflated Sharpe on the full-sample figure is 0.90 against a 0.95 gate — a near miss at only N=8.
A realistic search (N=48 for TSMOM) would deflate it considerably further. Neither buy-and-hold
control is beaten: BTC 1.05, ETH 1.29 annualised over the same window.

## 7. Caveats on these numbers

- The probe's return series has skew 61 and kurtosis 7,966. That is not a normal return
  distribution, and Sharpe is a poor summary of it. The DSR calculation uses those moments
  explicitly, which is why it is the reported statistic.
- Full-sample figures span train **and** validation. They are quoted only as the in-sample
  counterpart to the walk-forward number, never as a result.
- 370 capacity breaches on the ETH L24 run mean the reported path is one a $25k book could not
  fully trade. Turnover of 95×/yr against a 15 bps round-trip is a 1,427 bps/yr drag before any
  edge — the probe is not a serious strategy and its numbers should not be read as one.
