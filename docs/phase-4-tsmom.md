# Phase 4 — TSMOM verdict

**Date:** 2026-09-03 · **Data:** Binance BTC/ETH USD-M perpetuals, 2020-01-01 → 2024-06-30
(train + validation) · **Costs:** conservative (5.5 bps taker) · **N = 48** from the trial registry
· **Book:** $25,000 · **Sealed test period: not opened.**

## Verdict

**No configuration passed. 0 validated, 5 candidates, 43 killed.**

TSMOM is the only signal family in §2 with independent evidence of a net-of-fee edge, and this is
its first real test in this system. It came closer than anything else run here, and it did not
clear the gates.

## The headline configuration

```
STRATEGY   TSMOM_L168_H72_4h_BTCUSDT     PERIOD  2020-01-01 → 2024-06-30 (train+validation)
TRIALS N   48                            COSTS   conservative (5.5bps taker)
NET SHARPE 1.29    DEFLATED SHARPE 0.84  PBO     0.09
BREAKEVEN  104.1 bps round-trip
VERDICT    FAIL — deflated_sharpe 0.84 vs 0.95; fold_dispersion 1.90 vs 0.97;
                  max_drawdown 0.40 vs 0.35; parameter_plateau 0.06 vs 0.50
```

| Gate | Value | Threshold | |
|---|---:|---:|---|
| Net Sharpe (OOS) | 1.29 | > 1.00 | **pass** |
| Deflated Sharpe | 0.84 | > 0.95 | fail |
| PBO | 0.09 | < 0.30 | **pass** |
| Fold dispersion | 1.90 | < 0.97 | fail |
| Max drawdown | 40% | < 35% | fail |
| Break-even cost | 104.1 bps | > 30 bps | **pass** |
| Parameter plateau | 0.06 | ≥ 0.50 | fail |
| Beats controls | yes | — | **pass** |

This is a genuine near-miss, and worth reading carefully. It clears four gates including the two
that usually kill things: an out-of-sample Sharpe above 1, and a break-even cost of 104 bps against
a 15 bps round-trip — nearly 7× headroom. It also beats every control, including the
turnover-matched random-entry control that this project only just built.

**What stops it is the parameter plateau at 0.06.** Only 6% of the grid sits within 25% of the
peak. A real edge produces a region; this produces a spike. Combined with fold dispersion of 1.90 —
fold Sharpes scattered far wider than their mean — the honest reading is that one corner of the
grid happened to fit this particular four-year window. The deflated Sharpe agrees: 0.84 against a
0.95 gate, after correcting for all 48 trials.

## The bar-size split settles the horizon question

| Bar | Runs | Best OOS Sharpe | Median OOS Sharpe | Candidates |
|---|---:|---:|---:|---:|
| 1h | 24 | −0.14 | −0.73 | 0 |
| 4h | 24 | **+1.29** | −0.02 | 5 |

**The 1h arm is dead.** Not marginal — the best of 24 configurations is negative, and the median is
−0.73. Every surviving candidate is a 4h run.

This was [predicted in advance](horizon-analysis.md), before TSMOM was written, from the cost
arithmetic alone: at 1h a taker needs 72.5% directional accuracy to break even under conservative
costs, and at 4h it needs 61%. The prediction and the result were produced independently and they
agree.

## Where every configuration failed

| Gate | Failures |
|---|---:|
| Deflated Sharpe | 48/48 |
| Fold dispersion | 48/48 |
| Max drawdown | 48/48 |
| Parameter plateau | 48/48 |
| Beats controls | 47/48 |
| Net Sharpe (OOS) | 45/48 |
| Break-even cost | 36/48 |

Four gates failed universally. Break-even is the one most often cleared — 12 of 48 configurations
generate enough gross edge to survive their own trading costs, which is more than anything
previously run here managed. Costs are not what kills TSMOM. **Robustness is.**

## Long / short attribution (§8.1)

For the headline configuration:

| Leg | Bars | Time | Net P&L | Expectancy | Hit rate |
|---|---:|---:|---:|---:|---:|
| long | 5,382 | 54.6% | 218,524 | 79.8 bps | 51.5% |
| short | 4,281 | 43.5% | 21,040 | 7.6 bps | 49.4% |

The short leg earns roughly **one tenth** the expectancy of the long leg, at a hit rate below 50%.
It is barely paying for itself. This is exactly the asymmetry §8.1 requires be reported rather than
netted — momentum concentrates in winners while losers frequently rebound (Han, Kang & Ryu) — and
it means the headline number is carried almost entirely by the long side. A long-only variant would
be a different strategy and a new set of trials, not a free improvement.

## Tax (§17)

For the headline configuration: pre-tax +958%, post-tax +671% at an effective 30% rate, with TDS
withheld at **2.5× the account's capital per year**.

That TDS figure is a financing constraint, not a cost: §194S withholding is creditable against
final liability. But withholding two and a half times your capital annually and recovering it by
refund is a real operational problem, and it is the number a high-turnover strategy has to answer
for. The permanent effect is §115BBH: 30% on gains with no set-off and no carry-forward, so across
a grid containing losers the effective rate exceeds the headline rate.

## Method, fixed before any number was computed

- All 24 declared combinations registered across both symbols **before** results were computed;
  `N = 48` read back from the registry, never hard-coded.
- Targets computed once per (symbol, params) on the full series and **sliced** per fold. The
  Phase 1–3 harness regenerates inside each fold window, so its folds start cold — at 4h a 188-bar
  warm-up is a third of a 90-day fold. That pattern was deliberately not copied.
- PBO and the DSR trial dispersion computed **per bar size**. Stacking 1h and 4h return series
  measures the difference in observation frequency, not search dispersion.
- The random-entry control is matched on both exposure distribution and realised turnover, and the
  candidate is compared against its **95th percentile** rather than its mean — the candidate was
  itself selected as a maximum over 48 trials.
- The sealed test period was never opened and no token was requested. It remains available, once.

## What was fixed first, and why that ordering mattered

Five defects were corrected *before* TSMOM ran, because fixing any of them afterwards would be
indistinguishable from tuning until a gate passed:

1. `breakeven_cost_bps` returned a one-way figure while the gate compared it against twice a
   round-trip — demanding roughly 4× the true hurdle. Cost drag double-counted the same way.
2. `ewm_stdev` had no `min_samples`, so sigma came from two observations on early bars — and TSMOM
   levers *up* as sigma falls.
3. The shuffle test preserved drift, which a vol-scaled signal can harvest; it would have failed
   TSMOM as a false positive.
4. `cryptolab ingest --interval 4h` silently overwrote the stored 1h lake.
5. Two §11 gates — the matched random control and single-quarter concentration — were declared and
   never computed.

## What this does not say

It does not say TSMOM has no edge. It says **this** search, on **this** window, under conservative
costs, did not produce one that survives multiple-testing correction and walk-forward robustness.
The 4h/BTC corner is interesting enough to be worth a *pre-registered* re-test on the sealed period
one day — but that is the one touch this family ever gets, and spending it on a configuration that
fails four gates in-sample would be a waste of it.

Per §11.1, the killed configurations are not to be tuned. Per §12, all 48 reports are retained and
published.
