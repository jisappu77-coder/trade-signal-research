from __future__ import annotations

import polars as pl
import pytest

from cryptolab.data import quality
from cryptolab.data.quality import DataQualityError
from cryptolab.validation.synthetic import synthetic_bars


def test_clean_data_passes(bars):
    assert quality.check_ohlcv(bars, "BTCUSDT", "1h").passed


def test_missing_bars_fail_with_location_and_duration(bars):
    holed = pl.concat([bars.head(100), bars.tail(100)])
    report = quality.check_ohlcv(holed, "BTCUSDT", "1h")
    finding = next(f for f in report.findings if f.check == "missing_bars")
    assert not report.passed
    assert "bars missing" in finding.detail and "minutes" in finding.detail


def test_duplicate_timestamps_fail(bars):
    doubled = pl.concat([bars, bars.head(1)]).sort("open_time")
    report = quality.check_ohlcv(doubled, "BTCUSDT", "1h")
    assert any(f.check == "duplicate_timestamps" for f in report.findings)


def test_non_monotonic_timestamps_fail(bars):
    reversed_ = bars.reverse()
    report = quality.check_ohlcv(reversed_, "BTCUSDT", "1h")
    assert any(f.check == "non_monotonic" for f in report.findings)


def test_ohlc_violation_fails(bars):
    broken = bars.with_columns(
        pl.when(pl.arange(0, pl.len()) == 5).then(pl.col("low") * 0.5).otherwise(pl.col("high")).alias("high")
    )
    report = quality.check_ohlcv(broken, "BTCUSDT", "1h")
    assert any(f.check == "ohlc_violation" for f in report.findings)


def test_zero_volume_run_beyond_ten_bars_fails(bars):
    idx = pl.arange(0, pl.len())
    outage = bars.with_columns(
        pl.when(idx.is_between(50, 70)).then(0.0).otherwise(pl.col("volume")).alias("volume")
    )
    report = quality.check_ohlcv(outage, "BTCUSDT", "1h")
    assert any(f.check == "zero_volume_run" for f in report.findings)


def test_short_zero_volume_run_is_tolerated(bars):
    idx = pl.arange(0, pl.len())
    quiet = bars.with_columns(
        pl.when(idx.is_between(50, 54)).then(0.0).otherwise(pl.col("volume")).alias("volume")
    )
    assert quality.check_ohlcv(quiet, "BTCUSDT", "1h").passed


def test_unsupported_price_jump_fails():
    frame = synthetic_bars(200, seed=1)
    idx = pl.arange(0, pl.len())
    spiked = frame.with_columns(
        pl.when(idx == 100).then(pl.col("close") * 1.5).otherwise(pl.col("close")).alias("close")
    )
    report = quality.check_ohlcv(spiked, "BTCUSDT", "1h")
    assert any(f.check == "unsupported_price_jump" for f in report.findings)


def test_funding_cap_breach_fails():
    df = pl.DataFrame(
        {
            "funding_time": [0, 28_800_000],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "funding_rate": [0.0001, 0.02],
            "interval_hours": [8.0, 8.0],
            "mark_price": [20000.0, 20100.0],
        }
    )
    report = quality.check_funding(df, "BTCUSDT")
    assert any(f.check == "funding_cap_breach" for f in report.findings)


def test_report_hash_is_stable_and_sensitive(bars):
    a = quality.check_ohlcv(bars, "BTCUSDT", "1h")
    b = quality.check_ohlcv(bars, "BTCUSDT", "1h")
    c = quality.check_ohlcv(bars.head(500), "BTCUSDT", "1h")
    assert a.content_hash() == b.content_hash() != c.content_hash()


def test_failed_report_raises_with_detail(bars):
    report = quality.check_ohlcv(bars.reverse(), "BTCUSDT", "1h")
    with pytest.raises(DataQualityError, match="quality gate FAILED"):
        report.raise_if_failed()
