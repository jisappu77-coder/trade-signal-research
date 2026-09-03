"""The cost model — a fixed adversary, not a parameter (SPEC.md §7).

The four regimes are constants. There is no API on this module that lowers a fee, and adding one is
a protocol violation (CLAUDE.md non-negotiable #3). `conservative` is always the headline regime.

P&L is decomposed into `gross`, `fees`, `slippage` and `funding` because the §14.2.3 monotonicity
test asserts over fees+slippage only — funding is scaled by the regimes and a funding-*receiving*
position gets better as that multiplier rises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

Regime = Literal["optimistic", "expected", "conservative", "stressed"]

REGIME_ORDER: Final[tuple[Regime, ...]] = ("optimistic", "expected", "conservative", "stressed")
HEADLINE_REGIME: Final[Regime] = "conservative"

# §7 impact model: impact_bps = k * sqrt(order_notional / bar_quote_volume)
DEFAULT_IMPACT_K: Final[float] = 10.0
# Fills above this fraction of bar quote volume are rejected and logged as a capacity breach.
MAX_PARTICIPATION: Final[float] = 0.01

BPS: Final[float] = 1e-4


class CapacityBreachError(RuntimeError):
    """Raised when an order exceeds the §7 participation limit."""


@dataclass(frozen=True, slots=True)
class CostRegime:
    """One fixed cost regime. Frozen — instances are constants, not configuration."""

    name: Regime
    taker_fee_bps: float
    maker_fee_bps: float
    slippage_bps: float
    impact_multiplier: float
    funding_multiplier: float

    @property
    def round_trip_taker_bps(self) -> float:
        """Two taker fills' worth of fee plus slippage — the hurdle every signal must clear."""
        return 2.0 * (self.taker_fee_bps + self.slippage_bps)


REGIMES: Final[dict[Regime, CostRegime]] = {
    "optimistic": CostRegime("optimistic", 2.0, 0.0, 0.0, 0.0, 1.0),
    "expected": CostRegime("expected", 5.0, 2.0, 1.0, 0.0, 1.0),
    "conservative": CostRegime("conservative", 5.5, 2.0, 2.0, 1.0, 1.25),
    "stressed": CostRegime("stressed", 7.5, 3.0, 5.0, 2.0, 2.0),
}


def get_regime(name: str) -> CostRegime:
    """Look up a fixed regime by name. Unknown names raise rather than defaulting to something cheap."""
    if name not in REGIMES:
        raise KeyError(f"unknown cost regime {name!r}; the four fixed regimes are {list(REGIMES)}")
    return REGIMES[name]


@dataclass(frozen=True, slots=True)
class FillCost:
    """The cost of one fill, decomposed. All money amounts are in quote currency."""

    notional: float
    fee: float
    slippage: float
    impact: float
    half_spread: float
    participation: float

    @property
    def total(self) -> float:
        return self.fee + self.slippage + self.impact + self.half_spread

    @property
    def total_bps(self) -> float:
        return 0.0 if self.notional == 0 else self.total / self.notional / BPS


def impact_bps(order_notional: float, bar_quote_volume: float, k: float = DEFAULT_IMPACT_K) -> float:
    """§7 square-root impact. Zero-volume bars have infinite impact, so they are refused upstream."""
    if order_notional <= 0:
        return 0.0
    if bar_quote_volume <= 0:
        raise CapacityBreachError("cannot fill against a zero-quote-volume bar")
    return k * math.sqrt(order_notional / bar_quote_volume)


def fill_cost(
    order_notional: float,
    bar_quote_volume: float,
    regime: CostRegime,
    *,
    half_spread_bps: float = 0.0,
    maker: bool = False,
    k: float = DEFAULT_IMPACT_K,
    max_participation: float = MAX_PARTICIPATION,
) -> FillCost:
    """Cost of a single fill of `order_notional` (absolute value) against a bar.

    Raises CapacityBreachError above the participation limit — a breach is not silently absorbed
    into slippage, because a strategy that only works below its own capacity is not a strategy.
    """
    notional = abs(order_notional)
    if notional == 0:
        return FillCost(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    participation = notional / bar_quote_volume if bar_quote_volume > 0 else math.inf
    if participation > max_participation:
        raise CapacityBreachError(
            f"order notional {notional:,.0f} is {participation:.2%} of bar quote volume "
            f"{bar_quote_volume:,.0f}, above the {max_participation:.2%} limit (SPEC.md §7)"
        )

    fee_bps = regime.maker_fee_bps if maker else regime.taker_fee_bps
    imp_bps = regime.impact_multiplier * impact_bps(notional, bar_quote_volume, k)
    return FillCost(
        notional=notional,
        fee=notional * fee_bps * BPS,
        slippage=notional * regime.slippage_bps * BPS,
        impact=notional * imp_bps * BPS,
        half_spread=notional * half_spread_bps * BPS,
        participation=participation,
    )


def funding_payment(position_notional: float, funding_rate: float, regime: CostRegime) -> float:
    """Funding at one settlement, sign-aware (SPEC.md §7).

    Returns the cash flow *to the position*: negative is paid, positive is received. A long pays when
    funding is positive; a short receives it. The regime multiplier scales magnitude, which is why
    funding is exempt from the cost-monotonicity assertion (§14.2.3).
    """
    return -position_notional * funding_rate * regime.funding_multiplier


def breakeven_cost_bps(
    gross_return_per_turnover: float, turnover_per_year: float, *, tolerance: float = 1e-12
) -> float:
    """Round-trip cost in bps at which net return crosses zero (§7.1).

    `gross_return_per_turnover` is annualised gross return divided by annualised turnover, i.e. the
    gross edge available per unit of trading. Returns 0.0 when there is no gross edge to spend.
    """
    if turnover_per_year <= tolerance or gross_return_per_turnover <= 0:
        return 0.0
    return gross_return_per_turnover / BPS


def cost_drag_bps_per_year(turnover_per_year: float, round_trip_bps: float) -> float:
    """§9.4 — the arithmetic that decides whether a strategy can clear its own hurdle."""
    return turnover_per_year * round_trip_bps
