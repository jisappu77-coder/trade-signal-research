"""Return features (SPEC.md §8.1).

Every function here is causal by construction: row i uses rows <= i only. That is what the §14.2.1
shift test verifies, and it is why nothing in this module ever uses a centred or forward window.
"""

from __future__ import annotations

import polars as pl


def log_returns(close: str = "close", *, alias: str = "log_return") -> pl.Expr:
    """r_t = log(close_t / close_{t-1})."""
    return (pl.col(close) / pl.col(close).shift(1)).log().alias(alias)


def simple_returns(close: str = "close", *, alias: str = "simple_return") -> pl.Expr:
    return ((pl.col(close) / pl.col(close).shift(1)) - 1.0).alias(alias)


def momentum(lookback_bars: int, close: str = "close", *, alias: str | None = None) -> pl.Expr:
    """close_t / close_{t-L} - 1. The raw TSMOM input before vol-scaling."""
    if lookback_bars < 1:
        raise ValueError("lookback_bars must be >= 1")
    name = alias or f"momentum_{lookback_bars}"
    return ((pl.col(close) / pl.col(close).shift(lookback_bars)) - 1.0).alias(name)


def forward_return(horizon_bars: int, close: str = "close", *, alias: str | None = None) -> pl.Expr:
    """Forward return — **evaluation only**.

    This is the one non-causal expression in the package. It exists so attribution and hit-rate
    analysis can be written honestly; feeding it into a signal is a lookahead bug and the shift test
    will catch it.
    """
    name = alias or f"forward_return_{horizon_bars}"
    return ((pl.col(close).shift(-horizon_bars) / pl.col(close)) - 1.0).alias(name)
