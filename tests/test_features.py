from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cryptolab.backtest.engine import MS_PER_YEAR
from cryptolab.data.schemas import BAR_INTERVAL_MS
from cryptolab.features import derivatives, returns, volatility
from cryptolab.features.registry import FeatureRegistry, default_registry


def test_log_returns_match_numpy(bars):
    got = bars.select(returns.log_returns())["log_return"].to_numpy()[1:]
    close = bars["close"].to_numpy()
    # log(a/b) and log(a)-log(b) differ in the last bits; the shift test's bit-identical
    # requirement compares polars against polars, where this does not arise.
    np.testing.assert_allclose(got, np.diff(np.log(close)), rtol=1e-9)


def test_momentum_is_causal(bars):
    """Truncating the frame must not change earlier momentum values."""
    full = bars.select(returns.momentum(24))["momentum_24"].to_numpy()
    part = bars.head(500).select(returns.momentum(24))["momentum_24"].to_numpy()
    np.testing.assert_array_equal(part, full[:500])


def test_momentum_rejects_a_zero_lookback():
    with pytest.raises(ValueError, match="must be >= 1"):
        returns.momentum(0)


def test_forward_return_is_the_one_non_causal_expression(bars):
    """It exists for attribution only; truncation deliberately changes its tail."""
    full = bars.select(returns.forward_return(1))["forward_return_1"].to_numpy()
    part = bars.head(500).select(returns.forward_return(1))["forward_return_1"].to_numpy()
    assert np.isnan(part[-1]) and not np.isnan(full[499])


def test_ewm_stdev_is_causal(bars):
    frame = bars.select(returns.log_returns())
    full = frame.select(volatility.ewm_stdev(halflife=72))["sigma"].to_numpy()
    part = frame.head(600).select(volatility.ewm_stdev(halflife=72))["sigma"].to_numpy()
    np.testing.assert_allclose(part, full[:600], rtol=1e-12)


def test_annualise_carries_units_in_the_name(bars):
    frame = bars.select(returns.log_returns()).select(volatility.ewm_stdev(halflife=36))
    out = frame.select(volatility.annualise(bars_per_year=8766.0))
    assert out.columns == ["sigma_annualised"]


def test_drawdown_from_high_is_never_positive(bars):
    dd = bars.select(volatility.drawdown_from_high(window_bars=60))["drawdown_from_high"]
    assert dd.max() <= 1e-12


def test_funding_apr_annualises_by_interval():
    frame = pl.DataFrame({"funding_rate": [0.0001]})
    eight = frame.select(derivatives.funding_apr(interval_hours=8))["funding_apr"][0]
    four = frame.select(derivatives.funding_apr(interval_hours=4))["funding_apr"][0]
    assert eight == pytest.approx(0.0001 * 1095)
    assert four == pytest.approx(2 * eight)


def test_funding_apr_rejects_a_zero_interval():
    with pytest.raises(ValueError, match="must be positive"):
        derivatives.funding_apr(interval_hours=0)


def test_basis_uses_the_columns_it_is_given():
    frame = pl.DataFrame({"mark": [20_100.0], "index": [20_000.0]})
    assert frame.select(derivatives.basis_bps("mark", "index"))["basis_bps"][0] == pytest.approx(50.0)


def test_registry_rejects_duplicate_registration():
    registry = FeatureRegistry()
    registry.register("x", returns.log_returns, lookback_bars=1)
    with pytest.raises(KeyError, match="already registered"):
        registry.register("x", returns.log_returns, lookback_bars=1)


def test_registry_rejects_unknown_feature():
    with pytest.raises(KeyError, match="unknown feature"):
        FeatureRegistry().get("nope")


def test_max_lookback_is_the_largest_declared():
    registry = default_registry()
    assert registry.max_lookback(["log_return", "momentum"]) == 168


def test_registry_builds_features(bars):
    registry = default_registry()
    out = registry.build(bars, {"log_return": {}})
    assert "log_return" in out.columns


def test_ewm_stdev_suppresses_an_under_sampled_sigma():
    """A sigma from two observations is noise, and a vol-scaled signal turns it into leverage."""
    frame = pl.DataFrame({"log_return": [None, *np.random.default_rng(0).normal(0, 0.01, 40)]})
    sigma = frame.select(volatility.ewm_stdev(halflife=72, min_samples=20))["sigma"]
    assert sigma.null_count() == 20  # the null return plus 19 under-sampled bars
    assert sigma[25] is not None


def test_ewm_stdev_rejects_a_degenerate_min_samples():
    with pytest.raises(ValueError, match="at least two observations"):
        volatility.ewm_stdev(min_samples=1)


def test_bars_per_year_constants_cannot_drift():
    """The 4h constant and the engine's interval table are two sources of one truth."""
    assert pytest.approx(volatility.BARS_PER_YEAR_1H / 4) == volatility.BARS_PER_YEAR_4H
    assert pytest.approx(MS_PER_YEAR / BAR_INTERVAL_MS["4h"]) == volatility.BARS_PER_YEAR_4H
    assert pytest.approx(MS_PER_YEAR / BAR_INTERVAL_MS["1h"]) == volatility.BARS_PER_YEAR_1H
