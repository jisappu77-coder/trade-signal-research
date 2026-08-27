"""The §14.2 mandatory anti-lookahead tests.

These are written before the first real signal, per CLAUDE.md, and they run in CI on every signal.
The shift test and the shuffle test are the two that catch real bugs, so each is proven against
`LookaheadSignal` — a deliberately broken strategy — as well as against causal ones. A suite that
has never produced a true positive is not evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cryptolab.backtest.costs import REGIME_ORDER, get_regime
from cryptolab.backtest.engine import BacktestConfig, run_backtest
from cryptolab.backtest.portfolio import PortfolioState, apply_funding
from cryptolab.validation.synthetic import (
    LookaheadSignal,
    MomentumProbe,
    RandomSignal,
    ZeroSignal,
    synthetic_bars,
)

CAUSAL_SIGNALS = [ZeroSignal(), RandomSignal(seed=3), MomentumProbe()]


def _params(signal):
    return signal.grid()[0]


# ---- 1. shift test -------------------------------------------------------------------


@pytest.mark.parametrize("signal", CAUSAL_SIGNALS, ids=lambda s: s.name)
def test_shift_test_causal_signals_are_bit_identical(bars, signal):
    """Recompute with all future data truncated; outputs must be bit-identical (§14.2.1)."""
    params = _params(signal)
    vectorised = signal.generate(bars, params)

    for cut in (200, 500, 1200, bars.height):
        truncated = signal.generate(bars.head(cut), params)
        expected = vectorised.head(cut)
        assert truncated["timestamp"].to_list() == expected["timestamp"].to_list()
        np.testing.assert_array_equal(
            truncated["target_position"].to_numpy(),
            expected["target_position"].to_numpy(),
            err_msg=f"{signal.name} diverges when truncated at {cut} — lookahead bug",
        )


def test_shift_test_catches_a_real_lookahead_bug(bars):
    """The shift test must FAIL on a signal that peeks. This proves the test works."""
    signal = LookaheadSignal()
    vectorised = signal.generate(bars, {})
    truncated = signal.generate(bars.head(500), {})
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            truncated["target_position"].to_numpy(),
            vectorised.head(500)["target_position"].to_numpy(),
        )


# ---- 2. shuffle test -----------------------------------------------------------------


def _shuffled_bars(bars: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Rebuild a price path from the same returns in random order."""
    rng = np.random.default_rng(seed)
    close = bars["close"].to_numpy()
    rets = np.diff(close) / close[:-1]
    rng.shuffle(rets)
    new_close = close[0] * np.cumprod(1 + rets)
    new_close = np.concatenate([[close[0]], new_close])
    return bars.with_columns(
        pl.Series("close", new_close),
        pl.Series("open", np.concatenate([[close[0]], new_close[:-1]])),
        pl.Series("high", new_close * 1.001),
        pl.Series("low", new_close * 0.999),
    )


@pytest.mark.parametrize("signal", CAUSAL_SIGNALS, ids=lambda s: s.name)
def test_shuffle_test_sharpe_indistinguishable_from_zero(bars, signal):
    """On shuffled returns, a causal strategy must earn nothing (§14.2.2).

    Asserted on **gross** returns. The question this test asks is whether the signal has foresight,
    and net returns cannot answer it: under real costs every strategy loses on shuffled data, so a
    net-return version of this test would pass anything that merely trades enough.
    """
    config = BacktestConfig(regime_name="conservative", warmup_bars=200)
    sharpes = []
    for seed in range(6):
        shuffled = _shuffled_bars(bars, seed)
        targets = signal.generate(shuffled, _params(signal))
        result = run_backtest(shuffled, targets, config)
        sharpes.append(result.sharpe(gross=True))

    mean = float(np.mean(sharpes))
    stderr = float(np.std(sharpes, ddof=1)) / np.sqrt(len(sharpes)) if len(sharpes) > 1 else 0.0
    # Indistinguishable from zero: within 3 standard errors, and small in absolute terms over
    # 2000 bars. The cheating signal below clears this bar by an order of magnitude.
    assert abs(mean) < max(0.02, 3 * stderr), (
        f"{signal.name} earns {mean:.4f}/bar gross on shuffled data — lookahead"
    )


def test_shuffle_test_catches_a_real_lookahead_bug(bars):
    """The cheating signal must profit on shuffled data. This proves the test works."""
    config = BacktestConfig(regime_name="optimistic", warmup_bars=10, no_trade_band=0.0)
    # A short run against a deep book: perfect foresight compounds fast enough to outgrow any
    # book eventually, and the §7 capacity limit would fire before the test could make its point.
    shuffled = _shuffled_bars(synthetic_bars(200, seed=42, quote_volume=1e12), 0)
    targets = LookaheadSignal().generate(shuffled, {})
    result = run_backtest(shuffled, targets, config)
    assert result.sharpe(gross=True) > 0.05, (
        "the lookahead signal failed to cheat — the harness is broken"
    )


# ---- 3. cost monotonicity ------------------------------------------------------------


def test_cost_monotonicity_over_fees_and_slippage(bars):
    """Net-of-fee P&L is non-increasing as the regime worsens (§14.2.3).

    Asserted over gross - fees - slippage only. Funding is exempt: the regimes scale funding
    magnitude, and a funding-receiving position gets *better* as that multiplier rises, so
    asserting over total P&L would fail correct code.
    """
    signal = MomentumProbe()
    targets = signal.generate(bars, {"lookback": 24})

    # No funding series is supplied, so this run isolates fees and slippage exactly. The exemption
    # itself is covered by test_funding_regime_multiplier_scales_magnitude_only.
    net_returns, cost_bps = [], []
    for name in REGIME_ORDER:
        result = run_backtest(bars, targets, BacktestConfig(regime_name=name, warmup_bars=100))
        assert result.funding_paid == 0.0
        net_returns.append(result.total_return)
        traded = float(result.equity_curve["traded_notional"].sum())
        cost_bps.append(result.fees_and_slippage / traded if traded > 0 else 0.0)

    assert cost_bps == sorted(cost_bps), f"cost per unit traded not non-decreasing: {cost_bps}"
    assert net_returns == sorted(net_returns, reverse=True), (
        f"net return not non-increasing as costs worsen: {net_returns}"
    )


def test_gross_pnl_is_invariant_to_the_cost_regime(bars):
    """Gross P&L must not move with costs — if it does, costs are leaking into the price path."""
    signal = MomentumProbe()
    targets = signal.generate(bars, {"lookback": 48})
    results = [
        run_backtest(bars, targets, BacktestConfig(regime_name=n, warmup_bars=100, no_trade_band=1.5))
        for n in REGIME_ORDER
    ]
    # Sizing is equity-proportional, so a costlier run compounds from a smaller base and its
    # absolute gross P&L legitimately differs. What must not differ is the *position path*.
    paths = [r.equity_curve["target_position"].to_list() for r in results]
    assert all(p == paths[0] for p in paths), "cost regime changed the position path"
    assert all(r.gross_pnl >= r.net_pnl for r in results)


# ---- 4. zero-signal test -------------------------------------------------------------


@pytest.mark.parametrize("regime", REGIME_ORDER)
def test_zero_signal_produces_exactly_zero_pnl_and_zero_cost(bars, regime):
    """A constant-zero signal must produce exactly zero P&L and zero cost (§14.2.4)."""
    targets = ZeroSignal().generate(bars, {})
    result = run_backtest(bars, targets, BacktestConfig(regime_name=regime))
    assert result.net_pnl == 0.0
    assert result.fees_and_slippage == 0.0
    assert result.funding_paid == 0.0
    assert result.turnover_per_year == 0.0
    assert result.equity_curve["traded_notional"].sum() == 0.0


# ---- 5. funding sign test ------------------------------------------------------------


def test_long_through_positive_funding_loses_exactly_notional_times_rate():
    """A long through a positive settlement loses exactly notional x rate (§14.2.5)."""
    regime = get_regime("optimistic")  # funding_multiplier == 1.0
    state = PortfolioState(cash=0.0)
    state.position = type(state.position)(units=2.0, avg_price=20_000.0)
    price, rate = 20_000.0, 0.0001

    payment = apply_funding(state, price, rate, regime)

    assert payment == pytest.approx(-(2.0 * price * rate))
    assert state.funding_paid == pytest.approx(2.0 * price * rate)


def test_short_through_positive_funding_receives_exactly_notional_times_rate():
    """The mirror case — a short is *paid*. Sign errors here flatter every carry backtest."""
    regime = get_regime("optimistic")
    state = PortfolioState(cash=0.0)
    state.position = type(state.position)(units=-2.0, avg_price=20_000.0)

    payment = apply_funding(state, 20_000.0, 0.0001, regime)

    assert payment == pytest.approx(2.0 * 20_000.0 * 0.0001)
    assert state.funding_paid == pytest.approx(-(2.0 * 20_000.0 * 0.0001))


def test_funding_regime_multiplier_scales_magnitude_only():
    """Worse regimes scale funding magnitude; they never flip its sign."""
    state_long = PortfolioState(cash=0.0)
    state_long.position = type(state_long.position)(units=1.0, avg_price=20_000.0)
    paid = [
        abs(apply_funding(state_long, 20_000.0, 0.0001, get_regime(n))) for n in REGIME_ORDER
    ]
    assert paid == sorted(paid)


def test_funding_on_a_flat_position_is_zero():
    state = PortfolioState(cash=1000.0)
    assert apply_funding(state, 20_000.0, 0.01, get_regime("stressed")) == 0.0


# ---- engine discipline ---------------------------------------------------------------


def test_signal_at_bar_t_cannot_fill_at_bar_t(bars):
    """§9.1: a target stamped at bar t fills at bar t+1's open, never inside bar t."""
    small = synthetic_bars(10, seed=5)
    targets = pl.DataFrame(
        {
            "timestamp": small["open_time"],
            "target_position": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "confidence": [1.0] * 10,
        }
    )
    result = run_backtest(small, targets, BacktestConfig(regime_name="optimistic"))
    curve = result.equity_curve
    # The signal turns on at index 2, so the position may only appear from index 3.
    assert curve["position_units"][2] == 0.0
    assert curve["position_units"][3] != 0.0
