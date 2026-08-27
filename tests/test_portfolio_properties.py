"""§14.3 property tests: portfolio accounting invariants under `hypothesis`.

The invariant: cash + position value + realised costs = equity, always — for any sequence of
fills and funding settlements, at any prices.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from cryptolab.backtest.costs import REGIME_ORDER, get_regime
from cryptolab.backtest.portfolio import (
    PortfolioState,
    apply_fill,
    apply_funding,
    no_trade_band_filter,
)

prices = st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)
units = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
rates = st.floats(min_value=-0.0075, max_value=0.0075, allow_nan=False, allow_infinity=False)
regimes = st.sampled_from(REGIME_ORDER)

Operation = st.one_of(
    st.tuples(st.just("fill"), units, prices),
    st.tuples(st.just("funding"), rates, prices),
)


@settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
@given(ops=st.lists(Operation, min_size=1, max_size=25), regime_name=regimes)
def test_accounting_identity_holds_under_any_operation_sequence(ops, regime_name):
    """cash + position value == equity, and equity reconciles to costs and funding."""
    regime = get_regime(regime_name)
    initial = 1_000_000.0
    state = PortfolioState(cash=initial)
    book_depth = 1e12  # deep enough that no operation trips the capacity limit

    last_price = 20_000.0
    for kind, a, b in ops:
        if kind == "fill":
            assume(abs(a * b) > 1e-9)
            state, _ = apply_fill(state, a, b, book_depth, regime)
            last_price = b
        else:
            apply_funding(state, b, a, regime)
            last_price = b

    equity = state.equity(last_price)
    assert equity == pytest.approx(state.cash + state.position.notional(last_price), rel=1e-12)

    # Every bp charged left cash and landed in realised_costs / funding_paid.
    gross = equity - initial + state.realised_costs + state.funding_paid
    reconstructed = initial + gross - state.realised_costs - state.funding_paid
    assert equity == pytest.approx(reconstructed, rel=1e-9, abs=1e-6)


@settings(max_examples=200)
@given(delta=units, price=prices, regime_name=regimes)
def test_costs_are_never_negative(delta, price, regime_name):
    """A fill can never *pay* the strategy. A negative cost would be a rebate the spec doesn't grant."""
    assume(abs(delta * price) > 1e-9)
    state = PortfolioState(cash=1_000_000.0)
    _, cost = apply_fill(state, delta, price, 1e12, get_regime(regime_name))
    assert cost.fee >= 0 and cost.slippage >= 0 and cost.impact >= 0 and cost.total >= 0


@settings(max_examples=200)
@given(delta=units, price=prices)
def test_realised_costs_only_ever_increase(delta, price):
    assume(abs(delta * price) > 1e-9)
    state = PortfolioState(cash=1_000_000.0)
    before = state.realised_costs
    apply_fill(state, delta, price, 1e12, get_regime("conservative"))
    assert state.realised_costs >= before


@given(
    target=st.floats(-1.0, 1.0, allow_nan=False),
    current=st.floats(-1.0, 1.0, allow_nan=False),
    band=st.floats(0.0, 0.5, allow_nan=False),
)
def test_no_trade_band_suppresses_small_moves_entirely(target, current, band):
    """Below the band nothing trades; above it, the full delta trades (§9.3)."""
    delta = no_trade_band_filter(target, current, band)
    if abs(target - current) < band:
        assert delta == 0.0
    else:
        assert delta == pytest.approx(target - current)


@settings(max_examples=100)
@given(price=prices, rate=rates, unit=units)
def test_funding_is_symmetric_in_position_sign(price, rate, unit):
    """Flipping the position sign flips the funding cash flow exactly."""
    regime = get_regime("conservative")
    long_state = PortfolioState(cash=0.0)
    long_state.position = type(long_state.position)(units=unit, avg_price=price)
    short_state = PortfolioState(cash=0.0)
    short_state.position = type(short_state.position)(units=-unit, avg_price=price)

    assert apply_funding(long_state, price, rate, regime) == pytest.approx(
        -apply_funding(short_state, price, rate, regime)
    )


@settings(max_examples=100)
@given(unit=st.floats(0.1, 5.0), price=prices)
def test_a_closed_position_returns_to_flat(unit, price):
    state = PortfolioState(cash=1_000_000.0)
    apply_fill(state, unit, price, 1e12, get_regime("expected"))
    apply_fill(state, -unit, price, 1e12, get_regime("expected"))
    assert state.position.units == pytest.approx(0.0, abs=1e-12)
    # A round trip costs exactly two fills' worth and nothing more.
    assert state.cash == pytest.approx(1_000_000.0 - state.realised_costs, rel=1e-9)
