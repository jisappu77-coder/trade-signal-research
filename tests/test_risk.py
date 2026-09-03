from __future__ import annotations

from cryptolab.backtest.portfolio import DEFAULT_NO_TRADE_BAND
from cryptolab.backtest.risk import (
    KillReason,
    KillSwitch,
    RiskEngine,
    RiskLimits,
    audit_log_as_text,
)


def test_daily_loss_limit_trips():
    engine = RiskEngine(limits=RiskLimits(daily_loss_limit=0.05))
    engine.observe(0, 100_000)
    assert engine.check(1000, 94_000)
    assert engine.kill_switch.reason == KillReason.DAILY_LOSS


def test_drawdown_limit_trips():
    engine = RiskEngine(limits=RiskLimits(drawdown_limit=0.35, daily_loss_limit=0.99))
    engine.observe(0, 100_000)
    engine.observe(86_400_000, 60_000)
    assert engine.check(86_400_000, 60_000)
    assert engine.kill_switch.reason == KillReason.DRAWDOWN


def test_consecutive_losses_trip():
    engine = RiskEngine(limits=RiskLimits(max_consecutive_losses=3))
    engine.observe(0, 100_000)
    for _ in range(3):
        engine.record_trade_result(-100)
    assert engine.check(0, 100_000)
    assert engine.kill_switch.reason == KillReason.CONSECUTIVE_LOSSES


def test_a_win_resets_the_loss_streak():
    engine = RiskEngine(limits=RiskLimits(max_consecutive_losses=3))
    engine.record_trade_result(-100)
    engine.record_trade_result(-100)
    engine.record_trade_result(50)
    engine.observe(0, 100_000)
    assert not engine.check(0, 100_000)


def test_stale_data_trips():
    engine = RiskEngine(limits=RiskLimits(staleness_intervals=2))
    engine.observe(0, 100_000)
    assert engine.check(0, 100_000, bars_since_last_data=3)
    assert engine.kill_switch.reason == KillReason.STALE_DATA


def test_volatility_circuit_breaker_trips():
    engine = RiskEngine(limits=RiskLimits(volatility_circuit_multiple=4.0))
    engine.observe(0, 100_000)
    assert engine.check(0, 100_000, realised_vol=0.05, baseline_vol=0.01)
    assert engine.kill_switch.reason == KillReason.VOLATILITY


def test_a_healthy_book_does_not_trip():
    engine = RiskEngine()
    engine.observe(0, 100_000)
    assert not engine.check(0, 101_000, bars_since_last_data=1, realised_vol=0.01, baseline_vol=0.01)
    assert not engine.kill_switch.tripped


def test_clamp_zeroes_a_tripped_book():
    engine = RiskEngine()
    engine.kill_switch.trip(0, KillReason.MANUAL, "test")
    assert engine.clamp(1.0) == 0.0


def test_clamp_respects_the_position_cap():
    engine = RiskEngine(limits=RiskLimits(max_position_per_symbol=0.5))
    assert engine.clamp(1.0) == 0.5
    assert engine.clamp(-1.0) == -0.5


def test_first_reason_sticks():
    switch = KillSwitch()
    switch.trip(0, KillReason.DRAWDOWN, "first")
    switch.trip(1, KillReason.VOLATILITY, "second")
    assert switch.reason == KillReason.DRAWDOWN
    assert len(switch.audit_log) == 2


def test_reset_is_audited():
    switch = KillSwitch()
    switch.trip(0, KillReason.DRAWDOWN, "d")
    switch.reset(1)
    assert not switch.tripped
    assert switch.audit_log[-1].action == "reset"


def test_day_rollover_rebases_the_daily_limit():
    engine = RiskEngine(limits=RiskLimits(daily_loss_limit=0.05))
    engine.observe(0, 100_000)
    engine.observe(86_400_000, 90_000)  # new day: 90k is the new baseline
    assert not engine.check(86_400_000, 88_000)


def test_realised_leverage_breach_trips():
    engine = RiskEngine(limits=RiskLimits(max_gross_leverage=2.0))
    engine.observe(0, 100_000)
    assert engine.check(0, 100_000, gross_notional=1_000_000)
    assert engine.kill_switch.reason == KillReason.LEVERAGE


def test_a_fully_invested_book_at_the_cap_does_not_trip():
    """Sitting exactly at the limit is compliance, not a breach."""
    engine = RiskEngine(limits=RiskLimits(max_gross_leverage=2.0))
    engine.observe(0, 100_000)
    assert not engine.check(0, 100_000, gross_notional=200_000)


def test_leverage_tolerance_exceeds_the_no_trade_band():
    """Otherwise the §9.3 band alone kills every fully-invested book.

    The band lets exposure drift by `band` before rebalancing; if the risk engine trips inside that
    drift, a compliant strategy is stopped out for doing exactly what §9.3 tells it to do.
    """
    assert RiskLimits().leverage_tolerance > DEFAULT_NO_TRADE_BAND


def test_band_sized_drift_is_tolerated():
    engine = RiskEngine(limits=RiskLimits(max_gross_leverage=2.0))
    engine.observe(0, 100_000)
    drifted = 200_000 * (1 + DEFAULT_NO_TRADE_BAND)
    assert not engine.check(0, 100_000, gross_notional=drifted)


def test_insolvency_trips_ahead_of_everything_else():
    engine = RiskEngine()
    engine.observe(0, 100_000)
    assert engine.check(0, -5_000, gross_notional=1e9)
    assert engine.kill_switch.reason == KillReason.INSOLVENT


def test_audit_text_is_explicit_when_empty():
    assert audit_log_as_text([]) == "kill switch: never fired"


def test_audit_text_renders_entries():
    switch = KillSwitch()
    switch.trip(1_546_300_800_000, KillReason.DRAWDOWN, "drawdown -40%")
    assert "drawdown_limit" in audit_log_as_text(switch.audit_log)
