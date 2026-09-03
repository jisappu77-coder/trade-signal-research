from __future__ import annotations

import polars as pl
import pytest

from cryptolab.backtest.costs import CapacityBreachError
from cryptolab.backtest.engine import BacktestConfig, run_backtest
from cryptolab.backtest.risk import KillReason, RiskEngine, RiskLimits
from cryptolab.validation.synthetic import MomentumProbe, ZeroSignal, synthetic_bars


def constant_targets(bars: pl.DataFrame, value: float) -> pl.DataFrame:
    return bars.select(
        pl.col("open_time").alias("timestamp"),
        pl.lit(value).alias("target_position"),
        pl.lit(1.0).alias("confidence"),
    )


def test_a_backtest_needs_at_least_two_bars(bars):
    with pytest.raises(ValueError, match="at least two bars"):
        run_backtest(bars.head(1), constant_targets(bars.head(1), 0.0), BacktestConfig())


def test_buy_and_hold_tracks_the_price(bars):
    """A constant full-long target should earn roughly the price move.

    Risk limits are lifted here deliberately: with the §13 defaults the drawdown limit fires on this
    path and flattens the book, which is correct behaviour but a different test (see
    test_risk_engine_can_force_a_flatten).
    """
    permissive = RiskEngine(
        limits=RiskLimits(daily_loss_limit=1.0, drawdown_limit=1.0, max_consecutive_losses=10**9)
    )
    result = run_backtest(
        bars,
        constant_targets(bars, 1.0),
        BacktestConfig(regime_name="optimistic", half_spread_bps=0.0, max_leverage=1.0),
        risk=permissive,
    )
    price_move = bars["close"][-1] / bars["open"][1] - 1
    assert result.total_return == pytest.approx(price_move, rel=0.10)


def test_risk_defaults_flatten_a_severe_drawdown(bars):
    """With §13 defaults in place, the same path is stopped out rather than ridden down."""
    risk = RiskEngine()
    result = run_backtest(
        bars, constant_targets(bars, 1.0), BacktestConfig(regime_name="optimistic"), risk=risk
    )
    assert risk.kill_switch.tripped
    assert result.total_return > bars["close"][-1] / bars["open"][1] - 1


def test_no_trade_band_suppresses_turnover(bars):
    signal = MomentumProbe()
    targets = signal.generate(bars, {"lookback": 24})
    tight = run_backtest(bars, targets, BacktestConfig(no_trade_band=0.01, warmup_bars=100))
    loose = run_backtest(bars, targets, BacktestConfig(no_trade_band=1.5, warmup_bars=100))
    assert loose.turnover_per_year < tight.turnover_per_year


def test_turnover_and_cost_drag_are_reported(bars):
    result = run_backtest(
        bars, MomentumProbe().generate(bars, {"lookback": 24}), BacktestConfig(warmup_bars=100)
    )
    assert result.turnover_per_year > 0
    assert result.cost_drag_bps_per_year() == pytest.approx(
        result.turnover_per_year * result.config.regime.round_trip_taker_bps
    )


def test_warmup_bars_are_flat(bars):
    result = run_backtest(bars, constant_targets(bars, 1.0), BacktestConfig(warmup_bars=50))
    assert result.equity_curve["position_units"].head(50).abs().sum() == 0.0


def test_capacity_breach_rejects_the_fill_by_default():
    """§7: reject the fill and log it. The position stays put; the run continues."""
    thin = synthetic_bars(50, seed=1, quote_volume=1_000.0)
    result = run_backtest(thin, constant_targets(thin, 1.0), BacktestConfig())
    assert result.capacity_breaches > 0
    assert result.equity_curve["position_units"].abs().max() == 0.0
    assert result.equity_curve["traded_notional"].sum() == 0.0


def test_capacity_breach_can_be_made_fatal():
    thin = synthetic_bars(50, seed=1, quote_volume=1_000.0)
    with pytest.raises(CapacityBreachError):
        run_backtest(thin, constant_targets(thin, 1.0), BacktestConfig(on_capacity_breach="raise"))


def test_a_deep_book_reports_no_breaches(bars):
    result = run_backtest(bars, constant_targets(bars, 1.0), BacktestConfig(initial_equity=10_000.0))
    assert result.capacity_breaches == 0


def test_funding_is_applied_at_settlement_timestamps():
    bars = synthetic_bars(100, seed=2, vol=0.0)  # flat prices: funding is the only P&L
    settlements = bars["open_time"].to_list()[10:100:8]
    funding = pl.DataFrame(
        {
            "funding_time": settlements,
            "symbol": ["BTCUSDT"] * len(settlements),
            "funding_rate": [0.0005] * len(settlements),
            "mark_price": [20_000.0] * len(settlements),
        }
    )
    result = run_backtest(
        bars,
        constant_targets(bars, 1.0),
        BacktestConfig(regime_name="optimistic", half_spread_bps=0.0),
        funding=funding,
    )
    assert result.funding_paid > 0  # a long through positive funding pays
    assert result.equity_curve["funding_flow"].sum() < 0


def test_a_short_receives_funding():
    bars = synthetic_bars(100, seed=2, vol=0.0)
    settlements = bars["open_time"].to_list()[10:100:8]
    funding = pl.DataFrame(
        {
            "funding_time": settlements,
            "symbol": ["BTCUSDT"] * len(settlements),
            "funding_rate": [0.0005] * len(settlements),
            "mark_price": [20_000.0] * len(settlements),
        }
    )
    result = run_backtest(
        bars,
        constant_targets(bars, -1.0),
        BacktestConfig(regime_name="optimistic", half_spread_bps=0.0),
        funding=funding,
    )
    assert result.funding_paid < 0


def test_breakeven_cost_is_zero_without_a_gross_edge(bars):
    result = run_backtest(bars, ZeroSignal().generate(bars, {}), BacktestConfig())
    assert result.breakeven_cost_bps() == 0.0


def test_breakeven_cost_is_positive_for_a_profitable_path():
    """A signal with a real gross edge reports the cost at which it dies."""
    rising = synthetic_bars(500, seed=3, drift=0.002, vol=0.004)
    result = run_backtest(rising, constant_targets(rising, 1.0), BacktestConfig(regime_name="optimistic"))
    assert result.breakeven_cost_bps() > 0


def test_risk_engine_can_force_a_flatten():
    """A levered book in a crash breaches the leverage cap before the drawdown limit.

    Equity falls while the position does not, so realised leverage rises first. That ordering is
    deliberate: §13 evaluates limits most-severe-first, and running at 3x is a worse state to be in
    than being 10% down.
    """
    falling = synthetic_bars(400, seed=4, drift=-0.02, vol=0.01)
    risk = RiskEngine(limits=RiskLimits(drawdown_limit=0.99, daily_loss_limit=0.99, leverage_tolerance=0.02))
    result = run_backtest(
        falling,
        constant_targets(falling, 1.0),
        BacktestConfig(regime_name="optimistic", no_trade_band=0.01),
        risk=risk,
    )
    assert risk.kill_switch.tripped
    assert risk.kill_switch.reason == KillReason.LEVERAGE
    assert result.equity_curve["killed"].any()
    # Once killed, the book is flat and stays flat.
    assert result.equity_curve.filter(pl.col("killed"))["position_units"].abs().max() == 0.0


def test_unlevered_crash_trips_the_drawdown_limit_instead():
    """With no leverage to breach, the drawdown limit is the binding constraint."""
    falling = synthetic_bars(400, seed=4, drift=-0.02, vol=0.01)
    risk = RiskEngine(limits=RiskLimits(drawdown_limit=0.10, daily_loss_limit=0.99, max_gross_leverage=1.0))
    run_backtest(
        falling,
        constant_targets(falling, 0.5),
        BacktestConfig(regime_name="optimistic", max_leverage=1.0),
        risk=risk,
    )
    assert risk.kill_switch.reason == KillReason.DRAWDOWN


def test_a_book_that_cannot_delever_is_stopped_not_spiralled():
    """The defect this guards: a capacity-rejected reduction used to let leverage run away.

    Risk-reducing orders are now worked down to the participation cap, and the leverage limit is
    checked against realised exposure, so the book delevers or is killed — never both refused.
    """
    thin = synthetic_bars(300, seed=9, drift=-0.03, vol=0.02, quote_volume=2e6)
    risk = RiskEngine(limits=RiskLimits(daily_loss_limit=0.99, drawdown_limit=0.99))
    result = run_backtest(
        thin, constant_targets(thin, 1.0), BacktestConfig(initial_equity=50_000.0), risk=risk
    )
    assert result.equity_curve["exposure"].abs().max() < 10.0
    assert not result.went_bankrupt or result.equity_curve["equity"].min() > -1e-6


def test_insolvency_ends_the_run():
    """Trading on from negative equity produces a meaningless path, so the run stops.

    Equity-proportional sizing decays toward zero without crossing it, so insolvency needs a gap
    bigger than 1/leverage in a single bar — exactly the case a backtest must not paper over.
    """
    flat = synthetic_bars(100, seed=11, vol=0.0)
    crashed = flat["close"].to_numpy().copy()
    crashed[50:] *= 0.3  # a -70% gap, deeper than a 2x book can absorb
    gap = flat.with_columns(
        pl.Series("close", crashed),
        pl.Series("open", [flat["open"][0], *crashed[:-1]]),
        pl.Series("high", crashed * 1.001),
        pl.Series("low", crashed * 0.999),
    )
    risk = RiskEngine(
        limits=RiskLimits(
            daily_loss_limit=1.0,
            drawdown_limit=1.0,
            max_gross_leverage=10**6,
            max_consecutive_losses=10**9,
        )
    )
    result = run_backtest(
        gap, constant_targets(gap, 1.0), BacktestConfig(regime_name="optimistic"), risk=risk
    )
    assert result.went_bankrupt
    assert result.bars < gap.height  # stopped early rather than running to the end


def test_kill_switch_writes_an_audit_entry():
    falling = synthetic_bars(400, seed=4, drift=-0.02, vol=0.01)
    risk = RiskEngine(limits=RiskLimits(drawdown_limit=0.10, daily_loss_limit=0.99))
    run_backtest(falling, constant_targets(falling, 1.0), BacktestConfig(regime_name="optimistic"), risk=risk)
    assert risk.kill_switch.audit_log
    assert risk.kill_switch.audit_log[0].action == "flatten"


def test_summary_reports_every_header_field(bars):
    summary = run_backtest(
        bars, MomentumProbe().generate(bars, {"lookback": 24}), BacktestConfig(warmup_bars=100)
    ).summary()
    for key in (
        "net_pnl",
        "sharpe_per_bar",
        "max_drawdown",
        "turnover_per_year",
        "breakeven_cost_bps",
        "cost_drag_bps_per_year",
        "capacity_breaches",
    ):
        assert key in summary
