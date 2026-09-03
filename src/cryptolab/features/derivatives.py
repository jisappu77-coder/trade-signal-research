"""Derivatives features: funding, basis and open interest (SPEC.md §5.1, §8.3).

**Basis uses mark price, never last-traded price** (§5.1). `basis_bps` takes a mark-price column and
will not silently fall back to `close`.
"""

from __future__ import annotations

import polars as pl

HOURS_PER_YEAR = 8760.0


def funding_apr(
    funding_rate: str = "funding_rate", *, interval_hours: float, alias: str = "funding_apr"
) -> pl.Expr:
    """funding_rate x (8760 / interval_hours). Interval is derived from data, never assumed (§5.1)."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    return (pl.col(funding_rate) * (HOURS_PER_YEAR / interval_hours)).alias(alias)


def basis_bps(mark_price: str, index_price: str, *, alias: str = "basis_bps") -> pl.Expr:
    """Perp basis in bps from **mark** price against index. Never pass a last-traded price here."""
    return (((pl.col(mark_price) / pl.col(index_price)) - 1.0) * 1e4).alias(alias)


def oi_change(oi: str = "oi_quote", *, lookback_bars: int = 24, alias: str | None = None) -> pl.Expr:
    """Fractional change in open interest.

    Only usable where the daily collector has actually run — Binance retains ~30 days and history
    cannot be backfilled (§5.1).
    """
    return ((pl.col(oi) / pl.col(oi).shift(lookback_bars)) - 1.0).alias(alias or f"oi_change_{lookback_bars}")
