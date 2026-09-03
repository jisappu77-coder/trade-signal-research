"""TSMOM — volatility-scaled time-series momentum (SPEC.md §8.1).

The first Tier-1 signal in this project, and the only family in §2 with independent evidence of a
net-of-fee edge (Zarattini et al. 2025). Low turnover is the point: at ~11 bps round-trip, a signal
must not trade often, and the §9.3 no-trade band is what decides whether this clears its own costs.

    r_t      = log(close_t / close_{t-1})
    sigma_t  = EWM stdev of r over halflife H, annualised
    signal_t = sign(close_t / close_{t-L} - 1)
    target_t = signal_t * clip(sigma_target / sigma_t, 0, max_lev)

**Leverage convention.** §8.1's `target_t` runs to `max_lev` (2.0), but `Signal.validate_output`
requires `[-1, 1]` because leverage belongs to the portfolio layer. These agree exactly rather than
approximately: the engine defines `target_position == 1.0` as `max_leverage x equity` of notional
(see the step 4-6 comment in `backtest/engine.py`), so emitting `target_t / max_lev` and running
with `BacktestConfig.max_leverage = max_lev` reproduces §8.1's notional-to-equity ratio bar for bar.

**Known caveat, encoded rather than assumed away.** Momentum is concentrated in winners and losers
frequently rebound (Han, Kang & Ryu), so §8.1 requires the long and short legs be reported
separately. `backtest.attribution` computes that, and the report states plainly when the short leg
has negative expectancy net of costs instead of netting it into a single number.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import polars as pl

from cryptolab.data.schemas import BAR_INTERVAL_MS
from cryptolab.features import returns, volatility
from cryptolab.features.resample import infer_interval_ms
from cryptolab.signals.base import FeatureSpec, ParamRange, Signal

# Family constants, not searched axes. Adding either to `param_space` would change the registry's
# params key and re-register all 24 combinations as new trials (§10.4).
SIGMA_TARGET: Final[float] = 0.40
MAX_LEVERAGE: Final[float] = 2.0

BARS_PER_YEAR: Final[dict[str, float]] = {
    "1h": volatility.BARS_PER_YEAR_1H,
    "4h": volatility.BARS_PER_YEAR_4H,
}

# Longest declared lookback across the grid: L = 168 bars, plus the vol warm-up.
_MAX_LOOKBACK: Final[int] = 168


class TSMOM(Signal):
    """Volatility-scaled time-series momentum, per §8.1."""

    name = "tsmom"
    tier = 1
    required_features: ClassVar[list[FeatureSpec]] = [
        FeatureSpec("close", _MAX_LOOKBACK),
        FeatureSpec("log_return", 1),
        FeatureSpec("sigma", volatility.DEFAULT_MIN_SAMPLES),
    ]
    # The *entire* declared search space (§8.1): 4 x 3 x 2 = 24 combinations. Registered per symbol,
    # so a two-asset universe is N = 48. Expanding this is a registrable event that lowers every
    # deflated Sharpe drawn from the registry.
    param_space: ClassVar[dict[str, ParamRange]] = {
        "lookback_bars": ParamRange("lookback_bars", (24, 48, 96, 168)),
        "vol_halflife": ParamRange("vol_halflife", (36, 72, 144)),
        "bar": ParamRange("bar", ("1h", "4h")),
    }

    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        """Compute target exposure. Causal: row i reads rows <= i only."""
        bar = str(params.get("bar", "1h"))
        lookback = int(params.get("lookback_bars", 96))
        halflife = int(params.get("vol_halflife", 72))
        if bar not in BARS_PER_YEAR:
            raise ValueError(f"unknown bar {bar!r}; TSMOM is declared at {sorted(BARS_PER_YEAR)}")

        # The frame must actually be the bar the params claim, or sigma is annualised by the wrong
        # factor and the vol scaling is silently wrong by 2x. Aggregate with `resample.to_bar` first.
        self._assert_bar_matches(features, bar)

        # Rechunk so the result cannot depend on how the frame was assembled. Data read back from
        # the lake spans many Parquet files and arrives multi-chunk; the test fixtures do not.
        frame = features.rechunk()

        bars_per_year = BARS_PER_YEAR[bar]
        computed = frame.select(
            pl.col("open_time"),
            returns.momentum(lookback).alias("_momentum"),
            pl.col("close").alias("_close"),
        ).with_columns(
            frame.select(returns.log_returns())["log_return"].alias("_r"),
        )
        sigma = (
            computed.select(volatility.ewm_stdev("_r", halflife=halflife))
            .select(volatility.annualise(bars_per_year=bars_per_year))["sigma_annualised"]
            .alias("_sigma")
        )
        computed = computed.with_columns(sigma)

        scale = (SIGMA_TARGET / pl.col("_sigma")).clip(0.0, MAX_LEVERAGE) / MAX_LEVERAGE
        target = (
            pl.when(pl.col("_momentum") > 0)
            .then(scale)
            .otherwise(pl.when(pl.col("_momentum") < 0).then(-scale).otherwise(0.0))
        )

        out = computed.select(
            pl.col("open_time").alias("timestamp"),
            # Warm-up bars carry a null momentum or sigma and become a flat target rather than a
            # NaN. sigma == 0 makes the ratio infinite, which clips to max_lev — but paired with a
            # zero momentum sign it would be 0 * inf, so the sign branch above never multiplies.
            target.fill_nan(0.0).fill_null(0.0).alias("target_position"),
            # Confidence is the vol-scaling actually applied: low when the market is wild and the
            # position has been cut, high when it is calm.
            scale.fill_nan(0.0).fill_null(0.0).alias("confidence"),
        )
        return self.validate_output(out)

    @staticmethod
    def _assert_bar_matches(features: pl.DataFrame, bar: str) -> None:
        """Refuse a frame whose spacing contradicts the `bar` parameter."""
        if features.height < 2:
            return
        actual = infer_interval_ms(features)
        expected = BAR_INTERVAL_MS[bar]
        if actual != expected:
            raise ValueError(
                f"TSMOM was given {actual // 60_000}m bars but params say bar={bar!r} "
                f"({expected // 60_000}m). Annualising with the wrong factor silently misscales "
                "every position; aggregate with features.resample.to_bar first."
            )

    @property
    def max_lookback_bars(self) -> int:
        """Warm-up the engine must discard: the longest momentum lookback plus the vol warm-up."""
        return _MAX_LOOKBACK + volatility.DEFAULT_MIN_SAMPLES
