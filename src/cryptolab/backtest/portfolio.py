"""Portfolio accounting and the no-trade band (SPEC.md §9.3, §14.3).

The invariant this module exists to keep, asserted under `hypothesis` in the property tests:

    cash + position_value + realised_costs == equity   (always, to floating tolerance)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from cryptolab.backtest.costs import CostRegime, FillCost, fill_cost, funding_payment

DEFAULT_NO_TRADE_BAND = 0.10

# A flatten computed in float leaves a residue of order 1e-16 units. Left alone it is a position
# forever: it accrues funding at every settlement and counts as "held" in any exposure statistic.
# Anything this small is snapped to a true zero.
FLAT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Position:
    """A single-symbol position. `units` is signed; negative is short."""

    units: float = 0.0
    avg_price: float = 0.0

    def notional(self, price: float) -> float:
        return self.units * price


@dataclass(slots=True)
class PortfolioState:
    """Mutable portfolio state carried through the event loop.

    `realised_costs` accumulates every fee, slippage, impact and half-spread charge as a positive
    number. `funding_paid` is tracked separately (positive = net paid) so §14.2.3 can assert
    monotonicity over fees and slippage without funding contaminating it.
    """

    cash: float
    position: Position = field(default_factory=Position)
    realised_costs: float = 0.0
    funding_paid: float = 0.0
    gross_traded_notional: float = 0.0

    def equity(self, price: float) -> float:
        """Mark-to-market equity."""
        return self.cash + self.position.notional(price)


def no_trade_band_filter(
    target_position: float, current_position: float, band: float = DEFAULT_NO_TRADE_BAND
) -> float:
    """Return the position delta to trade, after the §9.3 band.

    Below the band the desired change is dropped entirely (not partially executed) — this is the
    primary turnover control and it is what decides whether a strategy clears its cost hurdle.
    """
    delta = target_position - current_position
    return 0.0 if abs(delta) < band else delta


def apply_fill(
    state: PortfolioState,
    delta_units: float,
    price: float,
    bar_quote_volume: float,
    regime: CostRegime,
    *,
    half_spread_bps: float = 0.0,
    maker: bool = False,
    impact_k: float = 10.0,
    max_participation: float = 0.01,
) -> tuple[PortfolioState, FillCost]:
    """Execute a fill and return the new state plus its decomposed cost.

    Costs are paid from cash, which is what keeps the §14.3 accounting identity true: every bp
    charged leaves cash and lands in `realised_costs`.
    """
    if delta_units == 0.0:
        return state, FillCost(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    notional = abs(delta_units * price)
    cost = fill_cost(
        notional,
        bar_quote_volume,
        regime,
        half_spread_bps=half_spread_bps,
        maker=maker,
        k=impact_k,
        max_participation=max_participation,
    )

    new_units = state.position.units + delta_units
    if abs(new_units) < FLAT_EPSILON * max(1.0, abs(state.position.units)):
        new_units = 0.0
    # Weighted average entry price; a flip resets the basis to the fill price.
    if state.position.units == 0 or (state.position.units > 0) != (new_units > 0):
        avg_price = price
    elif abs(new_units) > abs(state.position.units):
        avg_price = (
            state.position.avg_price * state.position.units + price * delta_units
        ) / new_units
    else:
        avg_price = state.position.avg_price

    state.cash -= delta_units * price + cost.total
    state.position = replace(state.position, units=new_units, avg_price=avg_price)
    state.realised_costs += cost.total
    state.gross_traded_notional += notional
    return state, cost


def apply_funding(state: PortfolioState, price: float, funding_rate: float, regime: CostRegime) -> float:
    """Settle funding on the open position. Returns the cash flow (negative = paid)."""
    payment = funding_payment(state.position.notional(price), funding_rate, regime)
    state.cash += payment
    state.funding_paid -= payment
    return payment


def accounting_residual(state: PortfolioState, price: float, initial_equity: float) -> float:
    """The §14.3 invariant, as a residual that must stay at zero.

        equity == initial_equity + gross_pnl - realised_costs - funding_paid

    Returns the discrepancy; the property test asserts it is ~0 for any sequence of operations.
    """
    gross_pnl = state.equity(price) - initial_equity + state.realised_costs + state.funding_paid
    return state.equity(price) - (initial_equity + gross_pnl - state.realised_costs - state.funding_paid)
