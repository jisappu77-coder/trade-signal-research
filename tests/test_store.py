from __future__ import annotations

import polars as pl
import pytest

from cryptolab.data.store import ParquetStore, SplitProtocol, from_ms, to_ms
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.sealed import SealedPeriodError, TestPeriodToken


def test_to_ms_round_trips():
    assert from_ms(to_ms("2019-01-01")).year == 2019
    assert to_ms("2019-01-01") == 1_546_300_800_000


def test_write_partitions_by_year_month(store, bars):
    paths = store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    assert len(paths) >= 2
    assert all("year=2019" in str(p) or "year=2020" in str(p) for p in paths)


def test_round_trip_read(store, bars, train_ms):
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://x")
    got = store.read("ohlcv", exchange="binance", symbol="BTCUSDT", start=train_ms[0], end=train_ms[1])
    assert got.height == bars.height
    assert got["source_uri"].unique().to_list() == ["test://x"]
    assert got["ingested_at"].null_count() == 0


def test_read_filters_to_window(store, bars):
    store.write(bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")
    got = store.read("ohlcv", exchange="binance", symbol="BTCUSDT", start="2019-01-01", end="2019-01-02")
    assert 0 < got.height < bars.height


def test_missing_symbol_returns_empty_typed_frame(store):
    got = store.read("ohlcv", exchange="binance", symbol="NOPE", start="2019-01-01", end="2019-02-01")
    assert got.height == 0 and "ingested_at" in got.columns


def test_end_before_start_raises(store):
    with pytest.raises(ValueError, match="precedes start"):
        store.read("ohlcv", exchange="binance", symbol="BTCUSDT", start="2020-01-01", end="2019-01-01")


# ---- the sealed test period (§10.1) -------------------------------------------------


def test_sealed_read_without_token_is_refused(store):
    with pytest.raises(SealedPeriodError, match="sealed test period"):
        store.read("ohlcv", exchange="binance", symbol="BTCUSDT", start="2024-07-01", end="2024-08-01")


def test_window_straddling_the_seal_is_refused(store):
    with pytest.raises(SealedPeriodError):
        store.read("ohlcv", exchange="binance", symbol="BTCUSDT", start="2024-01-01", end="2025-01-01")


def test_pre_seal_window_needs_no_token(store):
    store.read("ohlcv", exchange="binance", symbol="BTCUSDT", start="2019-01-01", end="2024-06-30")


def test_forged_token_is_refused(tmp_path, splits):
    registry = TrialRegistry(tmp_path / "r.sqlite")
    store = registry.bind_store(ParquetStore(tmp_path / "data", splits))
    forged = TestPeriodToken("tsmom", "deadbeef", "0" * 64)
    with pytest.raises(SealedPeriodError, match="invalid, already spent"):
        store.read(
            "ohlcv",
            exchange="binance",
            symbol="BTCUSDT",
            start="2024-07-01",
            end="2024-08-01",
            token=forged,
        )
    registry.close()


def test_valid_token_opens_the_seal_exactly_once(tmp_path, splits, bars):
    registry = TrialRegistry(tmp_path / "r.sqlite")
    store = registry.bind_store(ParquetStore(tmp_path / "data", splits))
    sealed_bars = bars.with_columns(pl.col("open_time") + (to_ms("2024-07-01") - to_ms("2019-01-01")))
    store.write(sealed_bars, "ohlcv", exchange="binance", symbol="BTCUSDT", source_uri="test://")

    token = registry.issue_test_token("tsmom")
    got = store.read(
        "ohlcv",
        exchange="binance",
        symbol="BTCUSDT",
        start="2024-07-01",
        end="2024-12-31",
        token=token,
    )
    assert got.height > 0

    # The same token is now spent.
    with pytest.raises(SealedPeriodError, match="already spent"):
        store.read(
            "ohlcv",
            exchange="binance",
            symbol="BTCUSDT",
            start="2024-07-01",
            end="2024-12-31",
            token=token,
        )
    registry.close()


def test_second_token_for_the_same_family_is_refused(tmp_path):
    registry = TrialRegistry(tmp_path / "r.sqlite")
    registry.issue_test_token("tsmom")
    with pytest.raises(SealedPeriodError, match="one-touch resource"):
        registry.issue_test_token("tsmom")
    registry.close()


def test_token_from_another_registry_is_refused(tmp_path, splits):
    issuer = TrialRegistry(tmp_path / "a.sqlite")
    other = TrialRegistry(tmp_path / "b.sqlite")
    store = other.bind_store(ParquetStore(tmp_path / "data", splits))
    token = issuer.issue_test_token("tsmom")
    with pytest.raises(SealedPeriodError):
        store.read(
            "ohlcv",
            exchange="binance",
            symbol="BTCUSDT",
            start="2024-07-01",
            end="2024-08-01",
            token=token,
        )
    issuer.close()
    other.close()


def test_unbound_store_refuses_even_a_real_token(tmp_path, splits):
    registry = TrialRegistry(tmp_path / "r.sqlite")
    store = ParquetStore(tmp_path / "data", splits)  # deliberately not bound
    token = registry.issue_test_token("tsmom")
    with pytest.raises(SealedPeriodError, match="no verifier bound"):
        store.read(
            "ohlcv",
            exchange="binance",
            symbol="BTCUSDT",
            start="2024-07-01",
            end="2024-08-01",
            token=token,
        )
    registry.close()


def test_open_ended_seal_covers_the_far_future():
    splits = SplitProtocol.from_config(
        {
            "train": {"start": "2019-01-01", "end": "2022-12-31"},
            "validation": {"start": "2023-01-01", "end": "2024-06-30"},
            "test": {"start": "2024-07-01", "end": None},
        }
    )
    assert splits.touches_sealed(to_ms("2030-01-01"), to_ms("2031-01-01"))
    assert not splits.touches_sealed(to_ms("2019-01-01"), to_ms("2020-01-01"))
