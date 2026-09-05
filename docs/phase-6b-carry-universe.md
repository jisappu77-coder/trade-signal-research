# Phase 6b — CARRY across the whole universe

**Date:** 2026-09-04 (benchmark corrected 2026-09-05) · **Data:** Binance USD-M perpetuals *and*
spot, **192 symbols**, 2020-01-01 →
2024-06-30 (train + validation) · **Costs:** conservative (15 bps round trip) plus §7 square-root
impact sized per bar · **Book:** $25,000 · **Trials `N` (carry family): 81** ·
**Sealed test period: not opened.**

## Verdict

**Widening the universe from 2 symbols to 192 moved the return from 2.37% to 2.65% post-tax. It did
not change the answer.**

| | Phase 6 (BTC + ETH) | Phase 6b (192 symbols) |
|---|---:|---:|
| Best pre-tax APR | 3.44% | **3.86%** |
| Best post-tax APR | 2.37% | **2.65%** |
| Configurations beating a 7% FD post-tax (30% slab → **4.82%**) | 0 / 54 | **0 / 81** |
| Configurations profitable pre-tax | 46 / 54 (85%) | **60 / 81 (74%)** |
| Worst configuration | — | **−8.60% pre-tax** |

A 12% relative improvement against a hurdle that needs **1.8×**. And the distribution got *worse*:
a quarter of the configurations now lose money outright, where Phase 6 had 85% profitable.

> **Correction (2026-09-05).** The first version of this page compared a post-tax strategy return
> against a **pre-tax** 7% deposit, and taxed VDA gains at 30% without the 4% cess. Indian FD
> interest is taxed at the holder's slab rate, so the honest hurdle is **4.82% post-tax** at the top
> bracket — a 7% figure applies only to someone below the rebate threshold. The verdict is unchanged
> and the gap is real, but it is 1.8× rather than the 2.6× first published, and the closing section
> on remaining levers was too dismissive. Both are fixed below.
>
> The re-run also picked up the **19 symbols the funding-interval guard had been dropping**, taking
> the universe from 173 to 192. Worth stating plainly: it did **not** move the headline. The best
> configuration is unchanged at 3.86% pre-tax. The guard's bias was real in principle and turned out
> to be immaterial here.

**This was my recommendation, and the estimate behind it was wrong.** I projected roughly 8%
post-tax on the reasoning that a wider universe would lift both the share of time capital is
deployed and the funding rate captured. The first barely moved and the second moved a fifth as far
as projected. The section below is why — the reasoning was wrong in a way the data makes precise,
which is worth more than the number was.

## The arithmetic, and where the estimate broke

The sleeve's return decomposes as

```
book APR  =  funding captured  ×  capital efficiency  ×  deployment fraction  −  costs
```

| Term | Phase 6 (ETH) | Phase 6b (best) | Projected |
|---|---:|---:|---:|
| Funding APR captured while deployed | 10.95% | **12.91%** | ~20% |
| Capital efficiency | 0.67 | 0.67 | ~0.9 |
| Deployment fraction | 47% | **44.8%** | ~85% |
| Costs as a share of gross funding | 31% | **35%** | — |

Deployment did not rise at all, and capture rose by a sixth. Three reasons, each measurable:

**1. The threshold gates entry, not the ranking.** With 12 slots the sleeve used **5.37 on
average** — half the book sat in cash even with 192 candidates. Funding above the entry threshold
in twelve markets at once is rare. Cross-sectional ranking only bites when more symbols qualify
than there are slots, and that is the uncommon case, so the extra 190 symbols mostly changed
nothing.

**2. Deployment and capture move in opposite directions, and the only knob controls both.** The
holding period sets the entry threshold, and lowering it to keep more capital busy buys worse
episodes at exactly the same rate:

| Hold | Entry threshold | Mean deployment | Mean hit rate | Mean pre-tax APR |
|---|---:|---:|---:|---:|
| 1 day | 221.0% | 10.4% | 64.6% | **1.76%** |
| 3 days | 75.0% | 28.5% | 39.8% | **0.37%** |
| 7 days | 33.3% | 47.7% | 38.2% | **0.08%** |

Deployment triples and the return *falls to nothing*. This is the finding that closes the family:
there is no threshold that makes both terms large, because they are the same knob pointing in
opposite directions.

**3. Selecting on the highest funding is adverse selection.** Funding is highest where a market is
under stress, and stress is what liquidates a short perp. The episode hit rate fell from 92–100% on
BTC/ETH to **66%** here, with **32% of 293 episodes liquidated**. The sleeve is not finding better
opportunities in the tail of the cross-section; it is finding more dangerous ones. This is
structural, not a parameter choice — the ranking criterion *is* a stress detector.

Where the money came from, for the best configuration: **funding +6,358, basis +205, costs −2,230**.
Costs rose from 31% of gross funding to 35%, because 293 episodes pay 1,172 fills where 16 paid 64.

## The grid

81 configurations: holding period (1, 3, 7 days) × exit fraction (0, 0.25, 0.5) × slots (4, 8, 12)
× margin (20%, 50%, 100%). Registered before the run, per §10.4.

| Hold | Exit | Slots | Margin | Episodes | Hit rate | Liquidated | Deployed | Pre-tax | Post-tax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3d | 0.00 | 12 | 50% | 293 | 66% | 31.7% | 44.8% | 3.86% | **2.65%** |
| 7d | 0.00 | 12 | 50% | 406 | 53% | 28.3% | 55.3% | 3.63% | 2.50% |
| 7d | 0.25 | 12 | 50% | 431 | 52% | 26.7% | 52.7% | 3.62% | 2.49% |
| 3d | 0.00 | 8 | 50% | 204 | 64% | 32.4% | 46.7% | 3.61% | 2.49% |
| 1d | 0.00 | 4 | 50% | 55 | 76% | 40.0% | 23.3% | 3.60% | 2.47% |

Best by slot count: 4 slots 2.47%, 8 slots 2.49%, 12 slots 2.65%. Tripling the slot count buys
18 basis points — another reading of the same ceiling.

**The failure record** (§12: it is the product). The worst four configurations all pay 20% margin on
a 7-day hold with a 0.5 exit fraction: −8.60%, −8.20%, −7.20%, −4.71% pre-tax. Thin margin plus a
high exit threshold means unwinding into the same stress that moved the price against the short leg.

## Survivorship: handled, not assumed away

The universe is enumerated from the archive's own key listing (`data/universe.py`), not from a list
of symbols anyone remembers. That distinction is the whole methodology here: **a funding carry dies
precisely where markets collapse** — funding spikes hardest just before a delisting — so a universe
of today's survivors would remove the losses the strategy is most exposed to and inflate every
number above.

- 471 USDT symbols have both a perp and a spot archive; 192 have a complete perp + spot + funding
  month, and **all 192 ingest with all three legs** once the funding-interval guard is fixed (it was
  dropping 19, including SOL and FTT); 84 traded at least one episode in the best configuration.
- LUNA, FTT, SRM, RAY, ANC, WAVES, TOMO, REN, CVC, BTS, SC and others are **in** the universe for
  the months they traded, and then simply end.

## Two defects this phase found

**The funding-interval guard was deleting the highest-funding episodes.** §5.1 requires refusing an
unhandled contract spec change, and the ingest did that by dropping any month whose funding interval
changed — 19 symbols, including SOL. Then look at *when*: SOL and FTT both failed in **2022-11**,
the FTX collapse. A venue shortens its funding interval (8h → 4h → 2h → 1h) exactly when funding
runs extreme, as a risk control. So the guard was systematically deleting the episodes a carry sleeve
earns most from — a selection effect pointing the wrong way, hidden inside a correctness check.

Fixed by carrying `interval_hours` **per settlement** through the schema, so a mid-month change is
recorded and followed rather than refused. `funding_interval_hours` still raises for callers that
genuinely need one scalar; `funding_intervals` returns the set.

**The 8-hour annualisation was hard-coded.** `funding_rate * 1095` appeared in three places. It is
correct only while a symbol settles every 8 hours, and §5.1 explicitly warns that symbols move to 4h.
A 4h symbol pays twice as often, so its APR was being halved. `carry.funding_apr()` now annualises
at the cadence in force on each bar. BTC and ETH never changed cadence within a month, so Phase 6's
published numbers are unaffected — the bug bit only the symbols this phase added.

## The cost model was strengthened, not relaxed

Extending to thin markets on Phase 6's flat cost would have been a quiet subsidy to exactly the
markets where funding is highest. Two changes, both adding cost:

- **Per-bar §7 square-root impact** on both legs, sized against that bar's own quote volume, using
  the same fixed regime constants. Verified equal to `costs.fill_cost` to 1e-15.
- **§7's 1% participation limit refuses entry** where a slot would be too large for the bar. An
  *exit* is never refused — the position already exists and unwinding it reduces risk, the same
  precedent `engine._apply_capacity_limit` sets. Re-running Phase 6's configuration under the new
  model moved ETH's cost from 1,573 to 1,593 (+1.3%) and its APR from 3.44% to 3.42%, so the
  comparison above is like for like.

## A dead end worth recording

I tried estimating a per-symbol effective spread with **Corwin & Schultz (2012)**, the standard
high-low bid-ask estimator, to price thin markets properly. It ordered symbols correctly (BTC below
the alts) but its absolute level was roughly **12× too high** — 6.3 bps half-spread for BTC, whose
true figure is well under 1 bp — and it got *worse* at lower frequencies (23.7 bps at 4h, 67.4 bps
at 1d for BTC). The estimator separates spread from variance by assuming variance scales with time
and spread does not; crypto's volatility is far above the equity market it was calibrated on, so it
is measuring volatility. Discarded, and the §7 impact model used instead. Recorded here rather than
deleted, per §12.

## What this does and does not say

It does not say funding carry has no edge. It says that at retail size, on this venue, over this
window, under conservative costs and Indian VDA tax, **the return is capped near 3–4% pre-tax by a
trade-off between how often capital works and how good the opportunities are**, and widening the
universe does not escape it because it does not change that trade-off.

The levers that remain are all outside the strategy rule, and against the *corrected* hurdle they
are closer than the first version of this page claimed:

| | Pre-tax | Post-tax |
|---|---:|---:|
| As measured | 3.86% | 2.65% |
| + maker execution (costs are 34% of gross) | 4.85% | 3.33% |
| + cross-margin (capital efficiency 0.67 → 0.90) | 6.54% | **4.50%** |
| **Hurdle (7% FD, 30% slab)** | | **4.82%** |

Both levers together land **32 basis points short** of a risk-free deposit. So the honest closing
statement is not "the return is far too small" — it is that in the best realistic case the sleeve
*matches* a deposit while carrying a 32% episode liquidation rate, two-venue execution risk and the
operational load of running both. Paying that for no excess return is the reason not to run it. The
first version of this page got the right verdict from a wrong number, which is worth recording.

The sealed test period remains unopened, and no §11 gate has been passed by anything in this project.
