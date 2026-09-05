"""Volatility features (SPEC.md §8.1, §8.2)."""

from __future__ import annotations

from typing import cast

import polars as pl

BARS_PER_YEAR_1H = 8766.0
BARS_PER_YEAR_4H = 2191.5

# A volatility estimate from two observations is noise, and TSMOM levers *up* as sigma falls, so an
# under-sampled sigma produces near-max-leverage positions on the first bars of every run.
DEFAULT_MIN_SAMPLES = 20


def ewm_stdev(
    returns: str = "log_return",
    *,
    halflife: int = 72,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    alias: str = "sigma",
) -> pl.Expr:
    """EWM standard deviation of returns over `halflife` bars. Causal; no centring.

    `min_samples` suppresses the estimate until enough observations exist. Without it polars emits a
    number from the second observation onward, and a sigma computed from two samples is noise — which
    any vol-scaled signal turns into leverage.
    """
    if halflife < 1:
        raise ValueError("halflife must be >= 1")
    if min_samples < 2:
        raise ValueError("min_samples must be >= 2; a stdev needs at least two observations")
    return (
        pl.col(returns).ewm_std(half_life=halflife, ignore_nulls=True, min_samples=min_samples).alias(alias)
    )


def annualise(
    sigma: str = "sigma", *, bars_per_year: float = BARS_PER_YEAR_1H, alias: str | None = None
) -> pl.Expr:
    """Scale a per-bar stdev to annual units. Named with its units, per CLAUDE.md."""
    # polars types `Expr.__mul__` as returning Any; the cast keeps --strict honest without an ignore.
    scaled = cast("pl.Expr", pl.col(sigma) * bars_per_year**0.5)
    return scaled.alias(alias or f"{sigma}_annualised")


def realised_vol(returns: str = "log_return", *, window_bars: int, alias: str = "realised_vol") -> pl.Expr:
    """Rolling realised volatility — the §8.2 regime input."""
    return pl.col(returns).rolling_std(window_bars, min_samples=max(2, window_bars // 4)).alias(alias)


def drawdown_from_high(
    close: str = "close", *, window_bars: int, alias: str = "drawdown_from_high"
) -> pl.Expr:
    """Drawdown from the rolling high over `window_bars` — negative when below the high."""
    return ((pl.col(close) / pl.col(close).rolling_max(window_bars, min_samples=1)) - 1.0).alias(alias)
