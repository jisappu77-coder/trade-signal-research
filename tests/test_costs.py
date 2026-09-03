from __future__ import annotations

import math

import pytest

from cryptolab.backtest.costs import (
    BPS,
    REGIME_ORDER,
    REGIMES,
    CapacityBreachError,
    cost_drag_bps_per_year,
    fill_cost,
    funding_payment,
    get_regime,
    impact_bps,
)


def test_the_four_regimes_match_the_spec_table():
    assert list(REGIMES) == list(REGIME_ORDER)
    assert REGIMES["conservative"].taker_fee_bps == 5.5
    assert REGIMES["conservative"].funding_multiplier == 1.25
    assert REGIMES["optimistic"].maker_fee_bps == 0.0
    assert REGIMES["stressed"].slippage_bps == 5.0


def test_regimes_are_frozen():
    with pytest.raises(Exception):
        REGIMES["conservative"].taker_fee_bps = 1.0  # type: ignore[misc]


def test_taker_fees_are_non_decreasing_across_regimes():
    fees = [REGIMES[n].taker_fee_bps for n in REGIME_ORDER]
    assert fees == sorted(fees)


def test_conservative_round_trip_is_about_fifteen_bps():
    assert REGIMES["conservative"].round_trip_taker_bps == pytest.approx(15.0)


def test_unknown_regime_raises_rather_than_defaulting_cheap():
    with pytest.raises(KeyError, match="unknown cost regime"):
        get_regime("free")


def test_impact_follows_the_square_root_law():
    assert impact_bps(1_000_000, 100_000_000, k=10) == pytest.approx(10 * math.sqrt(0.01))
    doubled = impact_bps(4_000_000, 100_000_000, k=10)
    assert doubled == pytest.approx(2 * impact_bps(1_000_000, 100_000_000, k=10))


def test_zero_volume_bar_cannot_be_filled():
    with pytest.raises(CapacityBreachError, match="zero-quote-volume"):
        impact_bps(1000, 0.0)


def test_capacity_breach_above_one_percent_participation():
    with pytest.raises(CapacityBreachError, match=r"above the"):
        fill_cost(2_000_000, 100_000_000, get_regime("conservative"))


def test_fill_just_below_the_limit_is_allowed():
    cost = fill_cost(999_000, 100_000_000, get_regime("conservative"))
    assert cost.participation < 0.01 and cost.total > 0


def test_fee_component_matches_the_regime():
    regime = get_regime("conservative")
    cost = fill_cost(100_000, 1e9, regime, maker=False)
    assert cost.fee == pytest.approx(100_000 * regime.taker_fee_bps * BPS)
    maker = fill_cost(100_000, 1e9, regime, maker=True)
    assert maker.fee == pytest.approx(100_000 * regime.maker_fee_bps * BPS)
    assert maker.fee < cost.fee


def test_zero_notional_costs_nothing():
    cost = fill_cost(0.0, 1e9, get_regime("stressed"))
    assert cost.total == 0.0 and cost.total_bps == 0.0


def test_total_cost_is_non_decreasing_across_regimes():
    totals = [fill_cost(100_000, 1e9, REGIMES[n]).total for n in REGIME_ORDER]
    assert totals == sorted(totals)


def test_funding_sign_convention():
    regime = get_regime("optimistic")
    assert funding_payment(10_000, 0.0001, regime) == pytest.approx(-1.0)  # long pays
    assert funding_payment(-10_000, 0.0001, regime) == pytest.approx(1.0)  # short receives
    assert funding_payment(10_000, -0.0001, regime) == pytest.approx(1.0)  # long receives


def test_funding_extreme_matches_the_spec_worked_example():
    """A +0.51%/8h average on a $10k long is roughly $153/day (§7)."""
    regime = get_regime("optimistic")
    daily = sum(funding_payment(10_000, 0.0051, regime) for _ in range(3))
    assert daily == pytest.approx(-153.0, abs=0.5)


def test_cost_drag_arithmetic():
    """At ~11 bps round-trip, trading 10x/day needs ~110 bps/day of gross edge (§9.4)."""
    assert cost_drag_bps_per_year(10 * 365, 11.0) == pytest.approx(365 * 110.0)
