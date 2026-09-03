"""Bar aggregation (SPEC.md §8.1 evaluates TSMOM at 1h and 4h).

4h bars are built by aggregating the stored 1h lake rather than ingesting the native 4h archive.
Two reasons, in order of importance:

1. **The lake has no interval dimension.** `store.partition_dir` keys on
   `dataset/exchange/symbol/year/month`, and `ingest_klines` writes every interval into
   `dataset="ohlcv"`. Ingesting 4h would overwrite the 1h partitions month by month.
2. **The result is identical anyway.** Binance 4h buckets are UTC-aligned at 00/04/08/12/16/20 and
   1h buckets nest inside them exactly, so aggregation reproduces the native bars by construction
   rather than approximately.

Bar-close discipline is preserved: a 4h close *is* the fourth 1h close, and the engine still fills
at the next bar's open.
"""

from __future__ import annotations

import polars as pl

from cryptolab.data import schemas
from cryptolab.data.schemas import BAR_INTERVAL_MS


class ResampleError(ValueError):
    """Raised when a frame cannot be aggregated to the requested bar."""


def infer_interval_ms(bars: pl.DataFrame) -> int:
    """The modal spacing between consecutive bars, in milliseconds."""
    if bars.height < 2:
        raise ResampleError("need at least two bars to infer an interval")
    diffs = bars.sort("open_time")["open_time"].diff().drop_nulls()
    modal = diffs.mode()
    if modal.is_empty():
        raise ResampleError("could not infer a bar interval")
    return int(modal[0])


def to_bar(bars: pl.DataFrame, bar: str) -> pl.DataFrame:
    """Aggregate `bars` up to `bar` (e.g. "4h"), returning the canonical OHLCV schema.

    Aggregating *down* is impossible and refused; a request for the interval already held is a
    no-op. The target interval must be a whole multiple of the source, or buckets would straddle
    source bars and the close would no longer be a real close.
    """
    if bar not in BAR_INTERVAL_MS:
        raise ResampleError(f"unknown bar {bar!r}; known: {sorted(BAR_INTERVAL_MS)}")
    target = BAR_INTERVAL_MS[bar]
    source = infer_interval_ms(bars)

    if target == source:
        return schemas.validate(bars.sort("open_time"), "ohlcv")
    if target < source:
        raise ResampleError(
            f"cannot produce {bar} bars from {source // 60_000}m data — aggregation only goes up"
        )
    if target % source != 0:
        raise ResampleError(
            f"{bar} ({target} ms) is not a whole multiple of the source interval ({source} ms); "
            "buckets would straddle source bars and the close would not be a real close"
        )

    expected = target // source
    aggregated = (
        bars.sort("open_time")
        .with_columns(pl.col("open_time").cast(pl.Datetime("ms")).alias("_ts"))
        .group_by_dynamic("_ts", every=f"{target}ms", closed="left", label="left")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("quote_volume").sum(),
            pl.col("trades").sum(),
            pl.col("taker_buy_base").sum(),
            pl.col("taker_buy_quote").sum(),
            pl.col("close_time").last(),
            pl.len().alias("_n"),
        )
        # An incomplete bucket is not a real bar: its "close" is whichever source bar happened to
        # land last, not the close of the period. That arises at the ends of a truncated window and
        # wherever the source has a gap, and it silently corrupts any signal reading the close.
        .filter(pl.col("_n") == expected)
        .with_columns(pl.col("_ts").dt.epoch("ms").cast(pl.Int64).alias("open_time"))
        .drop("_ts", "_n")
    )
    return schemas.validate(aggregated, "ohlcv")
