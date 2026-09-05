"""TSMOM against its §8.1 specification."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cryptolab.features import returns, volatility
from cryptolab.features.resample import to_bar
from cryptolab.signals.tsmom import BARS_PER_YEAR, MAX_LEVERAGE, SIGMA_TARGET, TSMOM
from cryptolab.validation.synthetic import synthetic_bars

DEFAULTS = {"bar": "1h", "lookback_bars": 96, "vol_halflife": 72}


def spec_formula(bars: pl.DataFrame, lookback: int, halflife: int, bar: str) -> np.ndarray:
    """§8.1 transcribed independently, to check the implementation against the text itself."""
    close = bars["close"].to_numpy()
    sigma = (
        bars.select(returns.log_returns())
        .select(volatility.ewm_stdev(halflife=halflife))
        .select(volatility.annualise(bars_per_year=BARS_PER_YEAR[bar]))["sigma_annualised"]
        .to_numpy()
    )
    momentum = np.concatenate([np.full(lookback, np.nan), close[lookback:] / close[:-lookback] - 1])
    return np.sign(momentum) * np.clip(SIGMA_TARGET / sigma, 0, MAX_LEVERAGE)


# ---- the declared search space ------------------------------------------------------


def test_grid_is_exactly_the_twenty_four_declared_combinations():
    """§8.1: L(4) x H(3) x bar(2). Expanding this silently would deflate every Sharpe wrongly."""
    grid = TSMOM().grid()
    assert len(grid) == 24
    assert {p["lookback_bars"] for p in grid} == {24, 48, 96, 168}
    assert {p["vol_halflife"] for p in grid} == {36, 72, 144}
    assert {p["bar"] for p in grid} == {"1h", "4h"}


def test_family_constants_are_not_searched_axes():
    """sigma_target and max_lev are family constants; adding them would re-register all 24 trials."""
    assert set(TSMOM().param_space) == {"lookback_bars", "vol_halflife", "bar"}


def test_is_tier_one():
    assert TSMOM().tier == 1


# ---- the formula --------------------------------------------------------------------


@pytest.mark.parametrize("lookback", [24, 168])
@pytest.mark.parametrize("halflife", [36, 144])
def test_matches_the_spec_formula_exactly(bars, lookback, halflife):
    """`target x max_lev` must equal §8.1's `sign * clip(sigma_target/sigma, 0, max_lev)`."""
    out = TSMOM().generate(bars, {"bar": "1h", "lookback_bars": lookback, "vol_halflife": halflife})
    ours = out["target_position"].to_numpy() * MAX_LEVERAGE
    expected = np.nan_to_num(spec_formula(bars, lookback, halflife, "1h"))
    finite = np.isfinite(ours) & np.isfinite(expected)
    np.testing.assert_allclose(ours[finite], expected[finite], rtol=0, atol=0)


def test_leverage_normalisation_keeps_the_output_in_range(bars):
    """The signal emits [-1, 1]; the gearing lives in BacktestConfig.max_leverage."""
    out = TSMOM().generate(bars, DEFAULTS)
    assert out["target_position"].abs().max() <= 1.0


def test_direction_follows_the_momentum_sign(bars):
    out = TSMOM().generate(bars, DEFAULTS)
    momentum = bars.select(returns.momentum(96))["momentum_96"].to_numpy()
    target = out["target_position"].to_numpy()
    both = np.isfinite(momentum) & (target != 0)
    assert np.all(np.sign(target[both]) == np.sign(momentum[both]))


def test_position_shrinks_as_volatility_rises():
    """Vol scaling is the whole point: a wilder market gets a smaller position."""
    calm = synthetic_bars(1500, seed=5, vol=0.004)
    wild = synthetic_bars(1500, seed=5, vol=0.02)
    calm_target = TSMOM().generate(calm, DEFAULTS)["target_position"].abs().mean()
    wild_target = TSMOM().generate(wild, DEFAULTS)["target_position"].abs().mean()
    assert wild_target < calm_target


def test_leverage_is_capped_in_a_dead_calm_market():
    """As sigma tends to zero the ratio explodes; the clip is what stops it."""
    flat = synthetic_bars(600, seed=1, vol=0.0, drift=0.0005)
    out = TSMOM().generate(flat, DEFAULTS)
    assert out["target_position"].abs().max() <= 1.0
    assert not np.isnan(out["target_position"].to_numpy()).any()


def test_warmup_bars_are_flat_not_nan(bars):
    """Null momentum or sigma becomes a zero target, never a NaN that poisons the equity curve."""
    out = TSMOM().generate(bars, DEFAULTS)
    target = out["target_position"].to_numpy()
    assert not np.isnan(target).any()
    assert np.all(target[:96] == 0.0)


def test_confidence_reports_the_applied_vol_scaling(bars):
    out = TSMOM().generate(bars, DEFAULTS)
    assert out["confidence"].min() >= 0.0
    assert out["confidence"].max() <= 1.0


# ---- bar handling -------------------------------------------------------------------


def test_runs_on_4h_bars(bars_4h):
    out = TSMOM().generate(bars_4h, {"bar": "4h", "lookback_bars": 96, "vol_halflife": 72})
    assert out.height == bars_4h.height
    assert out["target_position"].abs().max() <= 1.0


def test_4h_annualises_with_the_4h_constant(bars_4h):
    ours = TSMOM().generate(bars_4h, {"bar": "4h", "lookback_bars": 96, "vol_halflife": 72})
    expected = np.nan_to_num(spec_formula(bars_4h, 96, 72, "4h"))
    got = ours["target_position"].to_numpy() * MAX_LEVERAGE
    finite = np.isfinite(got) & np.isfinite(expected)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=0, atol=0)


def test_a_bar_param_contradicting_the_data_is_refused(bars):
    """Annualising 4h data at the 1h factor misscales every position by two. Refuse it."""
    with pytest.raises(ValueError, match="params say bar="):
        TSMOM().generate(bars, {"bar": "4h", "lookback_bars": 96, "vol_halflife": 72})


def test_resampled_4h_is_accepted(bars):
    out = TSMOM().generate(to_bar(bars, "4h"), {"bar": "4h", "lookback_bars": 24, "vol_halflife": 36})
    assert out.height == bars.height // 4


def test_unknown_bar_is_refused(bars):
    with pytest.raises(ValueError, match="unknown bar"):
        TSMOM().generate(bars, {"bar": "3h", "lookback_bars": 96, "vol_halflife": 72})


def test_max_lookback_covers_momentum_and_vol_warmup():
    assert TSMOM().max_lookback_bars >= 168 + volatility.DEFAULT_MIN_SAMPLES
