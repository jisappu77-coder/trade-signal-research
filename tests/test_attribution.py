"""Long/short attribution (§8.1), controls (§11) and tax treatment (§17)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cryptolab.backtest.attribution import as_frame, attribute
from cryptolab.backtest.engine import BacktestConfig, run_backtest
from cryptolab.backtest.risk import RiskEngine, RiskLimits
from cryptolab.signals.tsmom import TSMOM
from cryptolab.validation import tax
from cryptolab.validation.controls import (
    buy_and_hold_sharpe,
    calibrate_flip_probability,
    random_targets,
    single_quarter_concentrated,
    summarise_controls,
)
from cryptolab.validation.synthetic import synthetic_bars

PERMISSIVE = RiskLimits(daily_loss_limit=1.0, drawdown_limit=1.0, max_consecutive_losses=10**9)


def run(bars, **overrides):
    params = {"bar": "1h", "lookback_bars": 96, "vol_halflife": 72}
    settings = {"regime_name": "conservative", "warmup_bars": 200, **overrides}
    config = BacktestConfig(**settings)
    return run_backtest(bars, TSMOM().generate(bars, params), config, risk=RiskEngine(limits=PERMISSIVE))


def constant(bars, value):
    return bars.select(
        pl.col("open_time").alias("timestamp"),
        pl.lit(value).alias("target_position"),
        pl.lit(1.0).alias("confidence"),
    )


# ---- the invariant that keeps attribution honest -------------------------------------


def test_legs_sum_to_the_runs_net_pnl(bars):
    """Any scheme that silently loses P&L breaks this. It is the whole point of the module."""
    result = run(bars)
    attribution = attribute(result)
    assert attribution.residual == pytest.approx(0.0, abs=1e-6)
    total = sum(leg.net_pnl for leg in attribution.legs.values())
    assert total == pytest.approx(result.net_pnl, rel=1e-9)


def test_invariant_holds_with_funding_applied(bars):
    """Funding must land on the pre-fill position, or the legs stop reconciling."""
    settlements = bars["open_time"].to_list()[::8]
    funding = pl.DataFrame(
        {
            "funding_time": settlements,
            "symbol": ["BTCUSDT"] * len(settlements),
            "funding_rate": [0.0003] * len(settlements),
            "mark_price": [20_000.0] * len(settlements),
        }
    )
    result = run_backtest(
        bars,
        TSMOM().generate(bars, {"bar": "1h", "lookback_bars": 96, "vol_halflife": 72}),
        BacktestConfig(regime_name="conservative", warmup_bars=200),
        funding=funding,
        risk=RiskEngine(limits=PERMISSIVE),
    )
    assert attribute(result).residual == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("regime", ["optimistic", "conservative", "stressed"])
def test_invariant_holds_under_every_cost_regime(bars, regime):
    assert attribute(run(bars, regime_name=regime)).residual == pytest.approx(0.0, abs=1e-6)


def test_a_long_only_run_puts_everything_on_the_long_leg(bars):
    result = run_backtest(
        bars,
        constant(bars, 1.0),
        BacktestConfig(regime_name="optimistic"),
        risk=RiskEngine(limits=PERMISSIVE),
    )
    attribution = attribute(result)
    assert attribution.short.bars == 0
    assert attribution.long.bars > 0
    assert attribution.residual == pytest.approx(0.0, abs=1e-6)


def test_a_short_only_run_puts_everything_on_the_short_leg(bars):
    result = run_backtest(
        bars,
        constant(bars, -1.0),
        BacktestConfig(regime_name="optimistic"),
        risk=RiskEngine(limits=PERMISSIVE),
    )
    attribution = attribute(result)
    assert attribution.long.bars == 0
    assert attribution.short.bars > 0


def test_a_flat_run_has_no_legs_and_no_pnl(bars):
    result = run_backtest(bars, constant(bars, 0.0), BacktestConfig())
    attribution = attribute(result)
    assert attribution.long.bars == 0 and attribution.short.bars == 0
    assert attribution.legs["flat"].net_pnl == pytest.approx(0.0, abs=1e-9)


def test_negative_short_leg_is_stated_not_netted(bars):
    """§8.1: if the short leg has negative expectancy the report must say so."""
    attribution = attribute(run(bars))
    line = attribution.summary_line()
    assert "long" in line and "short" in line
    if attribution.short.negative_expectancy:
        assert "loses money net of costs" in line
    else:
        assert "positive net expectancy" in line


def test_expectancy_is_per_unit_of_notional(bars):
    attribution = attribute(run(bars))
    leg = attribution.long
    if leg.traded_notional > 0:
        assert leg.net_expectancy_bps == pytest.approx(leg.net_pnl / leg.traded_notional * 1e4)


def test_missing_columns_are_refused(bars):
    result = run(bars)
    result.equity_curve = result.equity_curve.drop("units_before_fill")
    with pytest.raises(ValueError, match="missing"):
        attribute(result)


def test_frame_view_has_a_row_per_leg(bars):
    assert as_frame(attribute(run(bars))).height == 3


# ---- controls (§11) -----------------------------------------------------------------


def test_random_targets_match_the_exposure_distribution():
    exposure = np.array([0.0, 0.2, -0.2, 0.4, -0.4] * 40)
    targets = random_targets(200, exposure, 0.1, seed=1)
    assert set(np.round(np.abs(targets), 6)) <= set(np.round(np.abs(exposure[exposure != 0]), 6))


def test_random_targets_flip_more_often_at_higher_probability():
    exposure = np.array([0.3] * 100)
    rare = random_targets(500, exposure, 0.01, seed=2)
    often = random_targets(500, exposure, 0.5, seed=2)
    assert (np.diff(np.sign(often)) != 0).sum() > (np.diff(np.sign(rare)) != 0).sum()


def test_an_all_flat_candidate_yields_a_flat_control():
    assert np.all(random_targets(50, np.zeros(50), 0.2, seed=3) == 0.0)


def test_calibration_finds_the_flip_probability_matching_turnover():
    """The control has to trade as much as the candidate, or beating it proves nothing."""
    calibrated = calibrate_flip_probability(40.0, lambda p: 100.0 * p)
    assert calibrated == pytest.approx(0.4, abs=0.02)


def test_calibration_of_a_zero_turnover_candidate_is_zero():
    assert calibrate_flip_probability(0.0, lambda p: p) == 0.0


def test_controls_are_compared_on_the_tail_not_the_mean():
    """The candidate was selected as a maximum over trials, so it must clear the tail."""
    control = summarise_controls("random", [0.0, 0.2, 0.4, 0.6, 0.8, 1.2])
    assert control.mean_sharpe < control.p95_sharpe
    assert not control.beaten_by(control.mean_sharpe + 0.01)
    assert control.beaten_by(control.p95_sharpe + 0.01)


def test_empty_control_set_is_handled():
    assert summarise_controls("random", []).p95_sharpe == 0.0


def test_buy_and_hold_sharpe_is_positive_in_an_uptrend():
    rising = synthetic_bars(2000, seed=4, drift=0.0008, vol=0.006)
    assert buy_and_hold_sharpe(rising, periods_per_year=8766.0) > 0


# ---- §11.1 single-quarter concentration ---------------------------------------------

QUARTER_MS = 92 * 86_400_000


def test_a_single_quarter_carrying_everything_is_flagged():
    """Declared in GateInputs since the gates were written, and never computed until now."""
    times = [1_577_836_800_000 + i * QUARTER_MS for i in range(6)]
    returns = [0.001, 0.001, 5.0, 0.001, 0.001, 0.001]
    assert single_quarter_concentrated(times, returns)


def test_evenly_spread_performance_is_not_flagged():
    times = [1_577_836_800_000 + i * QUARTER_MS for i in range(6)]
    assert not single_quarter_concentrated(times, [0.5] * 6)


def test_a_single_quarter_of_data_is_not_flagged():
    """One quarter of history cannot be 'concentrated in one quarter' — there is nothing to compare."""
    times = [1_577_836_800_000 + i * 3_600_000 for i in range(100)]
    assert not single_quarter_concentrated(times, [0.01] * 100)


def test_all_losses_are_not_flagged():
    times = [1_577_836_800_000 + i * QUARTER_MS for i in range(4)]
    assert not single_quarter_concentrated(times, [-0.1] * 4)


# ---- §17 tax ------------------------------------------------------------------------


def test_a_profitable_run_pays_thirty_percent():
    outcome = tax.tax_single_run(
        pre_tax_pnl=100_000.0, traded_notional=1e6, initial_equity=25_000.0, years=4.5
    )
    assert outcome.tax_due == pytest.approx(30_000.0)
    assert outcome.post_tax_pnl == pytest.approx(70_000.0)
    assert outcome.effective_rate == pytest.approx(0.30)


def test_a_loss_earns_no_relief():
    """§115BBH: no set-off against other income, no carry-forward. A loss simply buys nothing."""
    outcome = tax.tax_single_run(
        pre_tax_pnl=-50_000.0, traded_notional=1e6, initial_equity=25_000.0, years=4.5
    )
    assert outcome.tax_due == 0.0
    assert outcome.unusable_losses == pytest.approx(50_000.0)
    assert "cannot be set off" in tax.summary_line(outcome)


def test_losers_do_not_subsidise_winners_across_a_grid():
    """The asymmetry that changes the economics: winners taxed in full, losers stranded."""
    outcome = tax.tax_portfolio(
        [100_000.0, -60_000.0], traded_notional=1e6, initial_equity=25_000.0, years=4.5
    )
    assert outcome.pre_tax_pnl == pytest.approx(40_000.0)
    assert outcome.tax_due == pytest.approx(30_000.0)  # 30% of the winner alone
    assert outcome.post_tax_pnl == pytest.approx(10_000.0)
    # An effective rate far above the headline 30% is the whole point.
    assert outcome.effective_rate > 0.30


def test_tds_is_withholding_on_disposals_not_a_cost_per_round_trip():
    """1% of the sell side. Modelling it as 1% of every fill would double it."""
    outcome = tax.tax_single_run(pre_tax_pnl=1_000.0, traded_notional=2e6, initial_equity=25_000.0, years=1.0)
    assert outcome.tds_withheld == pytest.approx(0.01 * 1e6)


def test_tds_relative_to_capital_surfaces_the_financing_problem():
    outcome = tax.tax_single_run(pre_tax_pnl=1_000.0, traded_notional=5e6, initial_equity=25_000.0, years=1.0)
    assert outcome.tds_as_multiple_of_capital == pytest.approx(1.0)
    assert "creditable, but financed" in tax.summary_line(outcome)


def test_summary_reports_pre_and_post_tax_separately():
    """§17 requires both, separately."""
    line = tax.summary_line(
        tax.tax_single_run(pre_tax_pnl=50_000.0, traded_notional=1e6, initial_equity=25_000.0, years=4.5)
    )
    assert "pre-tax" in line and "post-tax" in line


def test_the_statutory_rates_are_the_ones_the_spec_cites():
    assert tax.VDA_RATE == 0.30  # §115BBH
    assert tax.TDS_RATE == 0.01  # §194S, not §115BBH — the mis-citation §17 warns about
