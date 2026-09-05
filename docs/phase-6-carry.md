# Phase 6 — CARRY verdict

**Date:** 2026-09-04 · **Data:** Binance BTC/ETH USD-M perpetuals *and spot*, 2020-01-01 →
2024-06-30 (train + validation) · **Costs:** conservative (5.5 bps taker) · **Book:** $25,000 ·
**Sealed test period: not opened.**

## Verdict

**CARRY works and is not worth running.**

Those are two separate findings and both matter. Across 54 configurations — the 9 declared parameter
combinations × 2 symbols × 3 margin levels — **46 are profitable net of every cost**, with episode
hit rates of 50–100% against §8.3's ~40% prior. This is the first strategy in this project that
consistently makes money.

**And not one of the 54 beats an Indian fixed deposit after tax.**

| | |
|---|---|
| Configurations profitable post-tax | **46 / 54** |
| Configurations beating a 7% FD post-tax (30% slab → 4.82%) | **0 / 54** |
| Best pre-tax APR | 3.44% |
| Best post-tax APR | **2.37%** |

> **Correction (2026-09-05).** This page originally compared a post-tax strategy return against a
> **pre-tax** 7% deposit, and taxed VDA gains at 30% without the 4% Health & Education Cess. Both
> are fixed. Indian FD interest is taxed at the holder's slab rate, so the honest hurdle is **4.82%
> post-tax** at the top bracket, not 7%; and §115BBH's real bite is **31.2%**, not 30%. The verdict
> is unchanged — nothing clears either version of the bar — but the gap is **1.8×, not 2.6×**, and
> the phrase "about a third of a fixed deposit" was wrong: it is **roughly half** of one.

The best configuration earns roughly **half a risk-free deposit** (2.37% against 4.82%), while
carrying liquidation risk on the short leg, execution risk across two venues, and the operational
load of running both. That is the verdict, and the 100% hit rate is not a reason to soften it.

## The grid

Top configurations by post-tax APR:

| Symbol | Hold | Exit | Margin | Episodes | Hit rate | Liquidated | Pre-tax APR | Post-tax APR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETH | 7d | 0.00 | 50% | 16 | 100% | 50% | 3.44% | **2.37%** |
| ETH | 7d | 0.25 | 50% | 16 | 94% | 44% | 3.38% | 2.33% |
| ETH | 3d | 0.00 | 50% | 14 | 93% | 57% | 2.93% | 2.02% |
| ETH | 7d | 0.00 | 100% | 12 | 100% | 33% | 2.89% | 1.99% |
| BTC | 7d | 0.00 | 50% | 12 | 83% | 33% | 2.87% | 1.97% |
| BTC | 7d | 0.25 | 100% | 12 | 100% | 8% | 2.10% | 1.44% |

Post-tax figures are the unchanged pre-tax numbers at the corrected 31.2% effective rate; for a
profitable run the relationship is exactly `post = pre × 0.688`.

**Read the liquidation column beside the return column.** The highest-returning configurations
liquidate a third to a half of their episodes. Reporting a single headline number without it would
be misleading: at 20% margin the 7-day hold liquidates 45–71% of episodes. The only configuration
that liquidates rarely (8%) is also among the lowest-returning.

Where the money comes from, for the best configuration: **funding +5,135, basis +308, costs −1,573**.
Funding is essentially the whole return; basis convergence contributes about 6% of it; costs consume
roughly a quarter of the gross. There is no hidden second source of edge here.

## Why this family was worth testing at all

Every other family in this project needs directional accuracy, and the
[horizon analysis](horizon-analysis.md) shows a retail taker needs 61–92% accuracy to break even
intraday — a bar nobody clears. CARRY needs **none**: it is long spot and short perp of equal size,
so price direction cancels and the return is funding minus the cost of holding both legs.

That structural difference is why CARRY was the honest next test after
[TSMOM failed](phase-4-tsmom.md), and it is why the result is positive where TSMOM's was not. The
strategy is sound. The *return* is simply too small to pay for its risks.

## What §8.3 required, and what is modelled

- **Costs on both legs, both ways.** Four fills per episode. Nothing is discounted for being
  market-neutral.
- **Margin and liquidation distance on the short perp.** Modelled as **isolated margin with no
  top-up** — the conservative reading, and the one a retail operator running two venues actually
  faces. A breach ends the episode even though the pair was economically hedged: an exchange
  liquidates the leg, not the strategy.
- **A forced close costs more than a signalled one.** A liquidation pays an extra 50 bps beyond the
  ordinary exit. Charging only the normal cost made liquidations look nearly free.
- **Funding-flip risk.** Hysteresis: deploy above the entry threshold, hold until funding falls
  below the exit level, rather than riding a decaying rate down.
- **The ~40% prior.** Reported on every run. Realised hit rates of 50–100% sit well above it — the
  prior turns out to be pessimistic for a threshold-gated entry, which is itself worth recording.

## Two bugs the tests caught

**The liquidation penalty was charged to equity but not recorded on the episode.** The equity curve
and the episode ledger disagreed, so per-episode P&L understated the cost of a forced close while
the account balance did not. Caught by a test asserting a liquidated episode's exit cost exceeds its
entry cost.

**§7's funding multiplier made the *stressed* regime the most profitable one.** The regimes scale
funding magnitude, which is adverse for a position that *pays* funding and favourable for one that
*receives* it — and a carry sleeve receives by construction. Applied naively, assuming worse
conditions increased the return, which is not modelling anything. `_adverse_funding_factor` in
`backtest/carry.py` now inverts the multiplier for received funding, so stress always means
receiving less. The §7 constants are untouched; only the direction of application is made consistent.
This is the same asymmetry that exempts funding from the §14.2.3 monotonicity assertion.

## Data handling

Spot is a separate dataset in the lake (`spot_ohlcv`), kept apart from the perp series so neither
overwrites the other — the same partition-collision hazard that
[silently overwrote the 1h lake](phase-4-tsmom.md) when ingesting 4h bars.

Twelve of the 54 spot months fail the §6 quality gate with **exactly one missing hour each** — real
single-bar outages on Binance spot. `align_legs` **drops** the bars where either leg has no price and
counts them (**31 across the period**) rather than filling them. A carry position cannot be opened or
closed at a price that never traded, and forward-filling one would be exactly the quiet assumption §6
exists to prevent.

## Live watch

`cryptolab live-signal` evaluates the §8.3 entry condition against the market now and logs each
reading; `cryptolab watch-summary` summarises a completed watch. It reads **public GETs only** — no
API key, no signing, no authenticated endpoint — consistent with §0 and §18. It places no orders.

**Venue caveat.** The backtest is Binance; the live readout is OKX, because `api.binance.com` and
`fapi.binance.com` both return HTTP 451 from this host. Funding differs between venues — that
difference is itself the `XFUND` Tier-2 signal §2 keeps disabled — so a live OKX reading indicates
the strategy's *state*, not a continuation of the backtested Binance series.

### Result: 28 observations, zero fires

The watch ran from **2026-09-03 20:53 to 2026-09-04 17:11 UTC** (20.3 hours of wall clock, 14
readings per symbol) against a **33.29%** entry threshold for a 7-day hold under conservative costs.

| Symbol | Readings | Fired | Funding APR range | Mean | **Closest approach** |
|---|---:|---:|---|---:|---:|
| BTCUSDT | 14 | **0** | +1.44% to +5.70% | +3.89% | **−27.58%** |
| ETHUSDT | 14 | **0** | +4.01% to +10.95% | +7.32% | **−22.34%** |

**Nothing fired, and nothing came close.** The closest approach — ETH's opening +10.95% — was still
**22 percentage points** below the threshold, and funding *fell* over the window on both legs rather
than rising toward it. The basis was mildly **negative** throughout (−3.5 to −6.3 bps): the perp
traded a shade cheap to spot, which is the opposite side of the carry entry. That is consistent with
funding this low. There was no crowd paying to be long.

This is a real result, not a failed experiment. The informative number is the closest approach, and
it says the funding regime over this window did not pay enough to cover two legs of cost — which is
exactly what the backtest predicts should happen most of the time. The best backtested configuration
was deployed under half the period; a day sampled at random should be expected to find the sleeve in
cash, and it did.

**A single day proves nothing about the strategy.** Twenty hours is not a sample, and no live
observation here changes a backtested verdict. What the watch establishes is narrower and still
worth having: the readout works end to end against a live venue, its entry arithmetic agrees with
the backtest's, and it correctly declines to fire. Raw log: [`live_watch.jsonl`](live_watch.jsonl).

## What this does not say

It does not say cash-and-carry is unprofitable — it plainly is profitable, in 46 of 54
configurations. It says that at retail scale, under conservative costs, with isolated margin, and
under Indian VDA tax, the return does not compensate for the risk taken. A larger book with cross-
margin, maker rebates, or a lower tax rate would change the arithmetic; none of those are available
here, and assuming them would be the kind of quiet favour §7 exists to prevent.

The sealed test period remains unopened. Nothing here has passed the §11 gates, and §18 still gates
live wiring behind a gate pass.
