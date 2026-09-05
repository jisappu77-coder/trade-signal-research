"""Bar aggregation and the interval guard that protects the stored series."""

from __future__ import annotations

import polars as pl
import pytest

from cryptolab.data.store import IntervalMismatchError
from cryptolab.features.resample import ResampleError, infer_interval_ms, to_bar
from cryptolab.validation.synthetic import synthetic_bars


def test_infers_the_source_interval(bars, bars_4h):
    assert infer_interval_ms(bars) == 3_600_000
    assert infer_interval_ms(bars_4h) == 14_400_000


def test_one_bar_cannot_yield_an_interval():
    with pytest.raises(ResampleError, match="at least two bars"):
        infer_interval_ms(synthetic_bars(2).head(1))


def test_four_one_hour_bars_become_one_four_hour_bar(bars):
    out = to_bar(bars, "4h")
    assert out.height == bars.height // 4
    assert infer_interval_ms(out) == 14_400_000


def test_ohlc_aggregates_correctly(bars):
    """open = first, high = max, low = min, close = last — the close must be a real close."""
    out = to_bar(bars, "4h")
    first_four = bars.head(4)
    assert out["open"][0] == first_four["open"][0]
    assert out["close"][0] == first_four["close"][3]
    assert out["high"][0] == first_four["high"].max()
    assert out["low"][0] == first_four["low"].min()


def test_volume_and_trades_are_conserved(bars):
    out = to_bar(bars, "4h")
    for col in ("volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        assert out[col].sum() == pytest.approx(bars[col].sum(), rel=1e-9)


def test_buckets_are_utc_aligned(bars):
    """Binance 4h buckets sit at 00/04/08/12/16/20 UTC; 1h bars nest inside them exactly."""
    out = to_bar(bars, "4h")
    assert (out["open_time"] % 14_400_000 == 0).all()


def test_incomplete_buckets_are_dropped(bars):
    """A partial bucket's close is not the close of the period — it is not a real bar.

    This is the defect that made a resampled month disagree with the native archive: a truncated
    window left a trailing bucket holding one source bar, whose close was simply the wrong price.
    """
    truncated = bars.head(bars.height - 2)  # leaves a 2-bar trailing bucket
    out = to_bar(truncated, "4h")
    assert out.height == truncated.height // 4
    assert out["close"][-1] == truncated["close"][out.height * 4 - 1]


def test_a_gap_does_not_produce_a_short_bar():
    """A hole in the source must drop the affected bucket, not emit a bar built from fewer bars."""
    full = synthetic_bars(80, seed=2)
    holed = pl.concat([full.head(4), full.slice(6)])  # remove two bars from the second bucket
    out = to_bar(holed, "4h")
    buckets = {int(t) for t in out["open_time"]}
    assert int(full["open_time"][4]) not in buckets  # the bucket missing two bars is gone
    assert int(full["open_time"][0]) in buckets  # complete buckets either side survive
    assert int(full["open_time"][8]) in buckets


def test_same_interval_is_a_no_op(bars):
    assert to_bar(bars, "1h").equals(bars.sort("open_time"))


def test_aggregation_only_goes_up(bars_4h):
    with pytest.raises(ResampleError, match="only goes up"):
        to_bar(bars_4h, "1h")


def test_non_multiple_intervals_are_refused():
    """A target that does not divide evenly would straddle source bars, so the close would lie."""
    seven_minute = synthetic_bars(100, seed=3, interval_ms=420_000)
    with pytest.raises(ResampleError, match="not a whole multiple"):
        to_bar(seven_minute, "15m")


def test_unknown_bar_is_refused(bars):
    with pytest.raises(ResampleError, match="unknown bar"):
        to_bar(bars, "3h")


def test_output_conforms_to_the_canonical_schema(bars):
    out = to_bar(bars, "4h")
    assert out.columns == bars.columns
    assert out.schema["trades"] == pl.Int64


# ---- the guard ----------------------------------------------------------------------


def test_writing_a_different_interval_is_refused(store, bars):
    """`cryptolab ingest SYMBOL --interval 4h` would otherwise destroy the stored 1h series."""
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    with pytest.raises(IntervalMismatchError, match="would overwrite the stored series"):
        store.write(to_bar(bars, "4h"), "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")


def test_the_refused_write_leaves_the_data_intact(store, bars):
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    before = store.read(
        "ohlcv", exchange="binance", symbol="BTCUSDT", start="2019-01-01", end="2019-12-31"
    ).height
    with pytest.raises(IntervalMismatchError):
        store.write(to_bar(bars, "4h"), "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    after = store.read(
        "ohlcv", exchange="binance", symbol="BTCUSDT", start="2019-01-01", end="2019-12-31"
    ).height
    assert after == before


def test_the_guard_can_be_overridden_deliberately(store, bars):
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    store.write(
        to_bar(bars, "4h"),
        "ohlcv",
        exchange="binance",
        symbol="BTCUSDT",
        source_uri="test://",
        allow_interval_change=True,
    )


def test_the_same_interval_writes_freely(store, bars):
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://again")


def test_a_new_symbol_is_unaffected(store, bars):
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    store.write(to_bar(bars, "4h"), "ohlcv", exchange="binance", symbol="ETHUSDT", source_uri="t://")
