"""Long/short leg attribution (SPEC.md §8.1).

§8.1 requires the long and short legs of a momentum strategy be reported **separately**, because
momentum is concentrated in winners while losers frequently rebound (Han, Kang & Ryu). If the short
leg has negative expectancy net of costs, the report must say so rather than netting it into a
single flattering number.

**The attribution rule.** The components of a bar's P&L belong to different legs at a flip, so the
rule is written down rather than left implicit:

* **Mark-to-market P&L splits at the open.** Fills happen at the bar's open (§9.1), so the move
  from the previous close into this open is earned by the position carried in, and the move from
  this open to this close by the position left after the fill. Attributing the whole close-to-close
  move to either position loses P&L at every flip — which is precisely where the long-versus-short
  question is decided, and the error that first broke the invariant below.
* **Fill cost** at bar `i` belongs to the leg being *entered*, since that is the trade being paid
  for.
* **Funding** at bar `i` belongs to the pre-fill position: the §9.2 event loop settles funding at
  step 1, before the fill at step 7.

The invariant that keeps this honest is that the legs sum to the run's net P&L. Any scheme that
silently loses money breaks it, and a test asserts it to float tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from cryptolab.backtest.engine import MS_PER_YEAR, BacktestResult

Leg = Literal["long", "short", "flat"]
LEGS: tuple[Leg, ...] = ("long", "short", "flat")


@dataclass(frozen=True, slots=True)
class LegAttribution:
    """One leg's contribution to a run."""

    leg: Leg
    bars: int
    fraction_of_time: float
    gross_pnl: float
    fees_and_slippage: float
    funding_paid: float
    net_pnl: float
    traded_notional: float
    mean_bar_return: float
    sharpe_per_obs: float
    hit_rate: float

    @property
    def net_expectancy_bps(self) -> float:
        """Net P&L per unit of notional traded, in bps — the §8.1 "expectancy" for this leg."""
        if self.traded_notional <= 0:
            return 0.0
        return self.net_pnl / self.traded_notional * 1e4

    @property
    def negative_expectancy(self) -> bool:
        """True when this leg loses money net of costs. §8.1 requires this be stated, not netted."""
        return self.bars > 0 and self.net_pnl < 0

    def sharpe_annualised(self, interval_ms: int) -> float:
        return self.sharpe_per_obs * math.sqrt(MS_PER_YEAR / interval_ms)


@dataclass(slots=True)
class Attribution:
    """Every leg of one run, plus the reconciliation against the run's own P&L."""

    legs: dict[Leg, LegAttribution] = field(default_factory=dict)
    net_pnl: float = 0.0
    interval_ms: int = 3_600_000

    @property
    def long(self) -> LegAttribution:
        return self.legs["long"]

    @property
    def short(self) -> LegAttribution:
        return self.legs["short"]

    @property
    def residual(self) -> float:
        """Legs minus the run's net P&L. Must be zero to float tolerance."""
        return sum(leg.net_pnl for leg in self.legs.values()) - self.net_pnl

    @property
    def short_leg_is_a_drag(self) -> bool:
        return self.short.negative_expectancy

    def summary_line(self) -> str:
        """The §8.1 statement, computed rather than written by hand."""
        long_leg, short_leg = self.long, self.short
        verdict = (
            "the short leg loses money net of costs and is reported separately rather than netted "
            "into the headline"
            if short_leg.negative_expectancy
            else "both legs carry positive net expectancy"
        )
        return (
            f"long {long_leg.net_expectancy_bps:+.1f} bps/notional over {long_leg.bars:,} bars; "
            f"short {short_leg.net_expectancy_bps:+.1f} bps/notional over {short_leg.bars:,} bars "
            f"— {verdict}"
        )


def _leg_of(units: np.ndarray) -> np.ndarray:
    """Map signed position size to a leg label index: 0 long, 1 short, 2 flat."""
    return np.where(units > 0, 0, np.where(units < 0, 1, 2))


def attribute(result: BacktestResult) -> Attribution:
    """Decompose a run's P&L into long, short and flat legs.

    Requires the `units_before_fill` and `bar_cost` columns the engine records; reconstructing them
    from shifted cumulative columns is possible but drifts at flips, which is exactly where the
    attribution question is decided.
    """
    curve = result.equity_curve
    missing = [c for c in ("units_before_fill", "bar_cost", "fill_price") if c not in curve.columns]
    if missing:
        raise ValueError(f"equity curve is missing {missing}; re-run the backtest to record them")
    if curve.height == 0:
        return Attribution(net_pnl=0.0, interval_ms=result.interval_ms)

    close = curve["price"].to_numpy()
    open_ = curve["fill_price"].to_numpy()
    held = curve["units_before_fill"].to_numpy()
    bar_cost = curve["bar_cost"].to_numpy()
    funding = curve["funding_flow"].to_numpy()
    traded = curve["traded_notional"].to_numpy()
    units_after = curve["position_units"].to_numpy()
    equity = curve["equity"].to_numpy()

    # The position changes at the bar's open (§9.1), so a bar's mark-to-market splits in two: the
    # move from the previous close into this open is earned by the position carried in, and the
    # move from this open to this close by the position left after the fill. Attributing the whole
    # close-to-close move to either one loses P&L at every flip — which is exactly where the
    # long-versus-short question is decided.
    prev_close = np.concatenate([[open_[0]], close[:-1]])
    gross_before = held * (open_ - prev_close)
    gross_after = units_after * (close - open_)

    held_leg = _leg_of(held)  # carried-in P&L and funding
    entered_leg = _leg_of(units_after)  # post-fill P&L, and the cost of entering

    prev_equity = np.concatenate([[result.config.initial_equity], equity[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_return = np.nan_to_num((equity - prev_equity) / prev_equity, nan=0.0, posinf=0.0)

    legs: dict[Leg, LegAttribution] = {}
    for index, name in enumerate(LEGS):
        held_mask = held_leg == index
        entered_mask = entered_leg == index
        leg_gross = float(gross_before[held_mask].sum() + gross_after[entered_mask].sum())
        leg_fees = float(bar_cost[entered_mask].sum())
        # `funding_flow` is the cash flow to the position: negative means paid.
        leg_funding = -float(funding[held_mask].sum())
        leg_returns = bar_return[held_mask]
        sd = float(np.std(leg_returns, ddof=1)) if leg_returns.size > 1 else 0.0
        wins = float((leg_returns > 0).sum())
        legs[name] = LegAttribution(
            leg=name,
            bars=int(held_mask.sum()),
            fraction_of_time=float(held_mask.sum()) / curve.height,
            gross_pnl=leg_gross,
            fees_and_slippage=leg_fees,
            funding_paid=leg_funding,
            net_pnl=leg_gross - leg_fees - leg_funding,
            traded_notional=float(traded[entered_mask].sum()),
            mean_bar_return=float(np.mean(leg_returns)) if leg_returns.size else 0.0,
            sharpe_per_obs=float(np.mean(leg_returns)) / sd if sd > 0 else 0.0,
            hit_rate=wins / leg_returns.size if leg_returns.size else 0.0,
        )

    return Attribution(legs=legs, net_pnl=result.net_pnl, interval_ms=result.interval_ms)


def as_frame(attribution: Attribution) -> pl.DataFrame:
    """Tabular view for the report."""
    return pl.DataFrame(
        [
            {
                "leg": leg.leg,
                "bars": leg.bars,
                "fraction_of_time": leg.fraction_of_time,
                "gross_pnl": leg.gross_pnl,
                "fees_and_slippage": leg.fees_and_slippage,
                "funding_paid": leg.funding_paid,
                "net_pnl": leg.net_pnl,
                "net_expectancy_bps": leg.net_expectancy_bps,
                "hit_rate": leg.hit_rate,
            }
            for leg in attribution.legs.values()
        ]
    )
