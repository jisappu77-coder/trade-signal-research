# What survives costs, by holding period

**Date:** 2026-09-03 · **Data:** Binance USD-M perpetuals, 1-minute bars, 2024-03, public archive
(44,640 bars per symbol)

This exists so that a proposal to trade a sub-hourly strategy meets the arithmetic before anyone
writes signal code. It is a §12 record: the point is what does *not* work.

## Method

For each holding period, the median absolute price move measured from real 1-minute BTC and ETH
bars, aggregated by taking every *k*-th close. A taker who is right with probability `p` earns
`(2p − 1) × E|move|` per round trip, so break-even requires

```
p > ½ · (1 + cost / E|move|)
```

When `cost ≥ E|move|`, no value of `p` suffices — the round-trip cost exceeds the whole typical
move, and even a perfect directional call loses money. Costs are the fixed §7 regimes; the
conservative column is the headline, per §7.

## Result

| Horizon | Median move | Optimistic (4 bps) | Expected (12 bps) | **Conservative (15 bps)** | Stressed (25 bps) |
|---|---:|---:|---:|---:|---:|
| 1 min | 4.8 bps | 91.9% | **impossible** | **impossible** | **impossible** |
| 5 min | 10.4 bps | 69.2% | **impossible** | **impossible** | **impossible** |
| 15 min | 17.7 bps | 61.3% | 83.8% | **92.3%** | **impossible** |
| 30 min | 24.2 bps | 58.3% | 74.8% | **81.0%** | **impossible** |
| 1 hour | 33.3 bps | 56.0% | 68.0% | **72.5%** | 87.5% |
| 4 hour | 68.2 bps | 52.9% | 58.8% | **61.0%** | 68.3% |
| 1 day | 219.8 bps | 50.9% | 52.7% | **53.4%** | 55.7% |

For calibration: a good systematic strategy runs 52–55% directional accuracy. 60% is exceptional.
Nothing sustainable runs at 72%.

## What this rules out

- **Below 15 minutes: arithmetically dead** for a retail taker under anything but optimistic costs.
  This is not a research problem to be solved with a better signal; the cost exceeds the move.
- **15–30 minutes:** needs 81–92% accuracy. Not attainable.
- **1 hour:** needs 72.5%. Implausible.
- **4 hour:** needs 61%. Demanding, but the first horizon worth testing.
- **1 day:** needs 53.4%. Comfortably in the range a real edge could occupy.

This is the same conclusion the citation audit behind SPEC.md reached — §2 places sub-hourly work,
OBI/OFI as taker strategies, and liquidation-cascade hunting in Tier 3 — now confirmed
independently on this project's own data.

## What it predicted, and what happened

The analysis was run *before* TSMOM was written, and it predicted the 4h arm would dominate the 1h
arm. [The Phase 4 results](phase-4-tsmom.md) bear that out: every one of the five configurations
that survived to `candidate` is a 4h run, and the best 1h configuration lands far below the worst
surviving 4h one. The prediction was made in advance and is recorded here rather than
reconstructed afterwards.

## Caveats

- One month of data (2024-03). Median moves vary with the volatility regime; a calmer month makes
  every row worse and a wilder one makes them better. The *ordering* is robust because move size
  scales with the square root of horizon while cost per round trip does not scale at all.
- The model assumes a symmetric win/loss of `E|move|`. A strategy with asymmetric payoffs (small
  frequent losses, rare large wins) can clear a lower accuracy — but it must then make that
  asymmetry explicit and defend it, which is what the §11 gates are for.
- Maker execution changes the arithmetic materially and is exactly why §2 admits `OFI_MAKER` as a
  Tier-2 execution improvement rather than a Tier-3 alpha source. It requires a confirmed rebate
  tier.
