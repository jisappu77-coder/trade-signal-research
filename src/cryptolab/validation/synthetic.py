"""Synthetic strategies for exercising the harness (SPEC.md §15, Phase 3 acceptance).

Phase 3 is accepted when DSR, PBO, walk-forward, the trial registry and sealed-test enforcement all
work **on synthetic strategies** — before a single real signal exists. These are those strategies.

`LookaheadSignal` is deliberately broken: it peeks at the next bar. It exists so the §14.2 shift and
shuffle tests are proven to *catch* a lookahead bug, rather than merely passing on code that happens
to be correct. A test suite that has never seen a true positive is not evidence.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import polars as pl

from cryptolab.signals.base import FeatureSpec, ParamRange, Signal


def synthetic_bars(
    n_bars: int,
    *,
    seed: int = 0,
    start_ms: int = 1_546_300_800_000,  # 2019-01-01 UTC
    interval_ms: int = 3_600_000,
    drift: float = 0.0,
    vol: float = 0.01,
    price0: float = 20_000.0,
    quote_volume: float = 5e8,
) -> pl.DataFrame:
    """A deterministic OHLCV frame with a known return process. No I/O, no network."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, vol, n_bars)
    close = price0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[price0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 4, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 4, n_bars)))
    times = start_ms + np.arange(n_bars, dtype=np.int64) * interval_ms
    return pl.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n_bars, quote_volume / price0),
            "quote_volume": np.full(n_bars, quote_volume),
            "trades": np.full(n_bars, 1000, dtype=np.int64),
            "taker_buy_base": np.full(n_bars, quote_volume / price0 / 2),
            "taker_buy_quote": np.full(n_bars, quote_volume / 2),
            "close_time": times + interval_ms - 1,
        }
    )


class ZeroSignal(Signal):
    """Returns a constant zero target — the §14.2.4 zero-signal test subject."""

    name = "zero"
    tier = 3
    required_features: ClassVar[list[FeatureSpec]] = [FeatureSpec("close", 1)]
    param_space: ClassVar[dict[str, ParamRange]] = {}

    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        return self.validate_output(
            features.select(
                pl.col("open_time").alias("timestamp"),
                pl.lit(0.0).alias("target_position"),
                pl.lit(0.0).alias("confidence"),
            )
        )


class RandomSignal(Signal):
    """A seeded random target. Causal (it never reads prices) and expected to earn nothing."""

    name = "random"
    tier = 3
    required_features: ClassVar[list[FeatureSpec]] = [FeatureSpec("close", 1)]
    param_space: ClassVar[dict[str, ParamRange]] = {"seed": ParamRange("seed", tuple(range(8)))}

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        rng = np.random.default_rng(int(params.get("seed", self.seed)))
        targets = rng.choice([-1.0, 0.0, 1.0], size=features.height)
        return self.validate_output(
            features.select(pl.col("open_time").alias("timestamp")).with_columns(
                pl.Series("target_position", targets),
                pl.lit(0.5).alias("confidence"),
            )
        )


class LookaheadSignal(Signal):
    """**Deliberately broken.** Reads bar t+1's close to decide bar t's position.

    Never promote this, never fix it: its job is to fail the shift and shuffle tests so those tests
    are known to work. If this signal ever passes §14.2, the tests are broken, not the signal.
    """

    name = "lookahead_cheat"
    tier = 3
    required_features: ClassVar[list[FeatureSpec]] = [FeatureSpec("close", 1)]
    param_space: ClassVar[dict[str, ParamRange]] = {}

    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        return self.validate_output(
            features.select(
                pl.col("open_time").alias("timestamp"),
                pl.when(pl.col("close").shift(-1) > pl.col("close"))
                .then(1.0)
                .otherwise(-1.0)
                .fill_null(0.0)
                .alias("target_position"),
                pl.lit(1.0).alias("confidence"),
            )
        )


class MomentumProbe(Signal):
    """A minimal causal momentum probe used to exercise the harness end to end.

    This is **not** the Tier-1 TSMOM signal of §8.1 — that belongs to Phase 4, after this harness is
    green. It has no vol scaling, no regime overlay, and it is not promotable.
    """

    name = "momentum_probe"
    tier = 3
    required_features: ClassVar[list[FeatureSpec]] = [FeatureSpec("close", 96)]
    param_space: ClassVar[dict[str, ParamRange]] = {"lookback": ParamRange("lookback", (12, 24, 48, 96))}

    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        lookback = int(params.get("lookback", 24))
        return self.validate_output(
            features.select(
                pl.col("open_time").alias("timestamp"),
                pl.when(pl.col("close") > pl.col("close").shift(lookback))
                .then(1.0)
                .otherwise(-1.0)
                .fill_null(0.0)
                .alias("target_position"),
                pl.lit(0.5).alias("confidence"),
            )
        )
