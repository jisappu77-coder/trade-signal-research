"""CARRY against its §8.3 specification."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cryptolab.backtest.carry import (
    LIQUIDATION_FEE_BPS,
    SURVIVAL_PRIOR,
    align_legs,
    run_carry_backtest,
)
from cryptolab.signals.carry import CarrySignal, entry_threshold_apr, funding_apr

HOUR = 3_600_000
START = 1_577_836_800_000  # 2020-01-01 UTC


def make_legs(n: int = 400, *, basis_bps: float = 10.0, spot_drift: float = 0.0) -> pl.DataFrame:
    """Perp and spot series with a controlled basis, plus 8h funding settlements."""
    times = START + np.arange(n) * HOUR
    spot = 20_000.0 * np.cumprod(np.full(n, 1.0 + spot_drift))
    perp = spot * (1 + basis_bps * 1e-4)
    perp_frame = pl.DataFrame(
        {
            "open_time": times,
            "open": perp,
            "high": perp * 1.001,
            "low": perp * 0.999,
            "close": perp,
            "volume": np.full(n, 100.0),
            "quote_volume": np.full(n, 5e9),
            "trades": np.full(n, 1000, dtype=np.int64),
            "taker_buy_base": np.full(n, 50.0),
            "taker_buy_quote": np.full(n, 2.5e9),
            "close_time": times + HOUR - 1,
        }
    )
    spot_frame = perp_frame.with_columns(
        pl.Series("open", spot),
        pl.Series("close", spot),
        pl.Series("high", spot * 1.001),
        pl.Series("low", spot * 0.999),
    )
    return perp_frame, spot_frame, times


def make_funding(times: np.ndarray, rate: float) -> pl.DataFrame:
    settlements = times[::8]
    return pl.DataFrame(
        {
            "funding_time": settlements,
            "symbol": ["BTCUSDT"] * len(settlements),
            "funding_rate": [rate] * len(settlements),
            "interval_hours": [8.0] * len(settlements),
            "mark_price": [20_000.0] * len(settlements),
        }
    )


def aligned_for(rate: float, **kwargs) -> tuple[pl.DataFrame, int]:
    perp, spot, times = make_legs(**kwargs)
    return align_legs(perp, spot, make_funding(times, rate))


def with_apr(aligned: pl.DataFrame) -> pl.DataFrame:
    return aligned.with_columns((pl.col("funding_in_force") * 1095.0).alias("funding_apr"))


# ---- the §8.3 arithmetic -------------------------------------------------------------


def test_funding_apr_annualises_by_the_actual_interval():
    """§5.1: some symbols moved to 4h, so the interval comes from data and is never assumed."""
    assert funding_apr(0.0001, 8) == pytest.approx(0.0001 * 1095)
    assert funding_apr(0.0001, 4) == pytest.approx(2 * funding_apr(0.0001, 8))


def test_funding_apr_rejects_a_degenerate_interval():
    with pytest.raises(ValueError, match="must be positive"):
        funding_apr(0.0001, 0)


def test_entry_threshold_demands_more_for_a_shorter_hold():
    """Paying the same round trip to earn funding for a day needs a far higher rate than a week."""
    day = entry_threshold_apr(15.0, 1.0, 0.02)
    week = entry_threshold_apr(15.0, 7.0, 0.02)
    assert day > week > 0.02


def test_entry_threshold_charges_both_legs():
    """A cash-and-carry round trip pays the round-trip cost twice — one per leg."""
    cheap = entry_threshold_apr(10.0, 3.0, 0.0)
    dear = entry_threshold_apr(20.0, 3.0, 0.0)
    assert dear == pytest.approx(2 * cheap)


def test_entry_threshold_includes_the_margin_buffer():
    assert entry_threshold_apr(15.0, 3.0, 0.05) - entry_threshold_apr(15.0, 3.0, 0.0) == pytest.approx(0.05)


# ---- the signal ----------------------------------------------------------------------


def test_deployment_is_not_directional():
    """§8.3: this is a sleeve, so target_position is deployment in [0, 1], never negative."""
    aligned, _ = aligned_for(0.003)
    out = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    assert out["target_position"].min() >= 0.0
    assert out["target_position"].max() <= 1.0


def test_high_funding_deploys_the_sleeve():
    aligned, _ = aligned_for(0.003)  # ~328% APR, far above any threshold
    out = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    assert out["target_position"].sum() > 0


def test_low_funding_never_deploys():
    aligned, _ = aligned_for(0.00001)  # ~1% APR, below every threshold
    out = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    assert out["target_position"].sum() == 0


def test_hysteresis_holds_through_a_dip():
    """Without hysteresis, funding oscillating about the threshold churns four fills for nothing."""
    aligned, _ = aligned_for(0.003)
    entry = entry_threshold_apr(15.0, 7.0, 0.02)
    apr = aligned["funding_in_force"].to_numpy() * 1095.0
    # Drop the rate to just under the entry threshold but above the exit level.
    dipped = apr.copy()
    dipped[100:150] = entry * 0.6
    frame = aligned.with_columns(pl.Series("funding_apr", dipped))
    out = CarrySignal().generate(frame, {"min_holding_days": 7.0, "exit_fraction": 0.25})
    assert out["target_position"][120] == 1.0  # still deployed through the dip


def test_falling_below_the_exit_level_closes_the_sleeve():
    """§8.3's funding-flip requirement: exit rather than ride the rate down."""
    aligned, _ = aligned_for(0.003)
    apr = aligned["funding_in_force"].to_numpy() * 1095.0
    apr[200:] = 0.0
    frame = aligned.with_columns(pl.Series("funding_apr", apr))
    out = CarrySignal().generate(frame, {"min_holding_days": 7.0, "exit_fraction": 0.25})
    assert out["target_position"][250] == 0.0


def test_missing_funding_column_is_refused():
    aligned, _ = aligned_for(0.001)
    with pytest.raises(ValueError, match="needs a funding_apr column"):
        CarrySignal().generate(aligned, {})


def test_the_declared_grid_is_nine_combinations():
    assert len(CarrySignal().grid()) == 9


def test_is_tier_one_and_named_carry():
    assert CarrySignal().tier == 1 and CarrySignal().name == "carry"


# ---- leg alignment -------------------------------------------------------------------


def test_a_missing_spot_bar_is_dropped_not_filled():
    """§6: inventing a price for a bar that did not trade is exactly the assumption to avoid."""
    perp, spot, times = make_legs(200)
    holed = pl.concat([spot.head(50), spot.slice(52)])  # two spot bars vanish
    aligned, dropped = align_legs(perp, holed, make_funding(times, 0.001))
    assert dropped == 2
    assert aligned.height == 198


def test_the_rate_in_force_is_the_last_settled_one():
    """A trader sees the last settled rate, not the next one — forward-fill, never back-fill."""
    aligned, _ = aligned_for(0.001)
    in_force = aligned["funding_in_force"].to_numpy()
    assert in_force[9] == pytest.approx(0.001)  # carried forward from the settlement at bar 8
    assert not np.isnan(in_force).any()


# ---- the backtest --------------------------------------------------------------------


def test_positive_funding_earns_the_sleeve_money():
    aligned, dropped = aligned_for(0.003)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets, dropped_bars=dropped)
    assert result.total_funding > 0
    assert result.net_pnl > 0


def test_four_fills_are_charged_per_episode():
    """Entry and exit on both legs — §8.3 is explicit that neither leg is free."""
    aligned, _ = aligned_for(0.003)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets)
    for episode in result.episodes:
        assert episode.entry_cost > 0 and episode.exit_cost > 0


def test_costs_rise_with_the_regime():
    aligned, _ = aligned_for(0.003)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    cheap = run_carry_backtest(aligned, targets, regime_name="optimistic")
    dear = run_carry_backtest(aligned, targets, regime_name="stressed")
    assert dear.total_costs > cheap.total_costs
    assert dear.net_pnl < cheap.net_pnl


def test_no_deployment_means_no_episodes_and_no_costs():
    aligned, _ = aligned_for(0.00001)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets)
    assert result.episodes == []
    assert result.total_costs == 0.0
    assert "never cleared the entry threshold" in result.prior_comparison()


def test_a_rising_perp_liquidates_the_short_leg():
    """The short is the leg that can be liquidated, even when the pair is economically hedged."""
    perp, spot, times = make_legs(400, spot_drift=0.0)
    rising = perp.with_columns(
        pl.Series("close", perp["close"].to_numpy() * np.linspace(1.0, 1.6, 400)),
        pl.Series("open", perp["open"].to_numpy() * np.linspace(1.0, 1.6, 400)),
    )
    aligned, _ = align_legs(rising, spot, make_funding(times, 0.003))
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets, margin_rate=0.20)
    assert result.liquidations > 0


def test_a_forced_close_costs_more_than_a_signalled_one():
    """Charging only the ordinary exit cost would make liquidations look almost free."""
    assert LIQUIDATION_FEE_BPS > 0
    perp, spot, times = make_legs(400)
    rising = perp.with_columns(
        pl.Series("close", perp["close"].to_numpy() * np.linspace(1.0, 1.6, 400)),
        pl.Series("open", perp["open"].to_numpy() * np.linspace(1.0, 1.6, 400)),
    )
    aligned, _ = align_legs(rising, spot, make_funding(times, 0.003))
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets, margin_rate=0.20)
    forced = [e for e in result.episodes if e.exit_reason == "liquidation"]
    assert forced and all(e.exit_cost > e.entry_cost for e in forced)


def test_more_margin_means_fewer_liquidations():
    perp, spot, times = make_legs(400)
    rising = perp.with_columns(
        pl.Series("close", perp["close"].to_numpy() * np.linspace(1.0, 1.6, 400)),
        pl.Series("open", perp["open"].to_numpy() * np.linspace(1.0, 1.6, 400)),
    )
    aligned, _ = align_legs(rising, spot, make_funding(times, 0.003))
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    thin = run_carry_backtest(aligned, targets, margin_rate=0.10)
    thick = run_carry_backtest(aligned, targets, margin_rate=1.00)
    assert thick.liquidations <= thin.liquidations


def test_the_hit_rate_is_reported_against_the_forty_percent_prior():
    """§8.3 requires exactly this comparison."""
    assert SURVIVAL_PRIOR == 0.40
    aligned, _ = aligned_for(0.003)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets)
    assert "prior in §8.3" in result.prior_comparison()
    assert 0.0 <= result.hit_rate <= 1.0


def test_pnl_decomposes_into_funding_basis_and_costs():
    aligned, _ = aligned_for(0.003)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets)
    reconstructed = result.total_funding + result.total_basis_pnl - result.total_costs
    assert reconstructed == pytest.approx(result.net_pnl, rel=1e-9)


def test_the_sleeve_is_delta_neutral_when_the_basis_is_flat():
    """A constant basis means price moves cancel: the only P&L left is funding minus costs."""
    aligned, _ = aligned_for(0.003, spot_drift=0.001)  # strong price trend, constant basis
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    result = run_carry_backtest(aligned, targets)
    # Basis P&L is negligible beside funding despite a large move in the underlying.
    assert abs(result.total_basis_pnl) < abs(result.total_funding)


def test_fills_happen_one_bar_after_the_signal():
    """§9.1 applies to both legs."""
    aligned, _ = aligned_for(0.003)
    targets = CarrySignal().generate(with_apr(aligned), {"min_holding_days": 7.0})
    first_signal = int(np.argmax(targets["target_position"].to_numpy() > 0))
    result = run_carry_backtest(aligned, targets)
    assert result.episodes[0].entry_time == int(aligned["open_time"][first_signal + 1])
