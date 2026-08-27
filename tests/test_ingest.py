from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import polars as pl
import pytest

from cryptolab.data.ingest import (
    months_between,
    open_interest_loss_notice,
    write_quality_reports,
)
from cryptolab.data.quality import check_funding, check_ohlcv
from cryptolab.data.sources import binance_api, binance_archive
from cryptolab.data.sources.binance_archive import ArchiveError, ArchiveObject
from cryptolab.data.store import to_ms


def _zip_of(csv: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("BTCUSDT-1h-2019-01.csv", csv)
    return buffer.getvalue()


ROW = "1546300800000,3701.23,3720.0,3695.0,3710.5,100.5,1546304399999,372000.0,1500,50.2,186000.0,0"


def test_archive_uri_matches_the_published_layout():
    obj = ArchiveObject("klines", "BTCUSDT", "1h", 2019, 1)
    assert obj.uri.endswith("klines/BTCUSDT/1h/BTCUSDT-1h-2019-01.zip")


def test_parse_klines_produces_the_canonical_schema():
    df = binance_archive.parse_klines(_zip_of(ROW), "test://")
    assert df.columns[:5] == ["open_time", "open", "high", "low", "close"]
    assert df["open_time"][0] == 1_546_300_800_000
    assert df.schema["trades"] == pl.Int64


def test_parse_klines_normalises_microsecond_timestamps():
    """Binance switched some 2025+ archives to microseconds. Magnitude decides, not assumption."""
    micro = ROW.replace("1546300800000", "1546300800000000").replace(
        "1546304399999", "1546304399999000"
    )
    df = binance_archive.parse_klines(_zip_of(micro), "test://")
    assert df["open_time"][0] == 1_546_300_800_000


def test_parse_klines_handles_a_header_row():
    header = (
        "open_time,open,high,low,close,volume,close_time,"
        "quote_volume,trades,taker_buy_base,taker_buy_quote,ignore"
    )
    df = binance_archive.parse_klines(_zip_of(f"{header}\n{ROW}"), "test://")
    assert df.height == 1


def test_a_corrupt_archive_raises():
    with pytest.raises(ArchiveError, match="not a valid zip"):
        binance_archive.parse_klines(b"not a zip", "test://")


def test_an_archive_with_two_csvs_raises():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("a.csv", ROW)
        zf.writestr("b.csv", ROW)
    with pytest.raises(ArchiveError, match="exactly one CSV"):
        binance_archive.parse_klines(buffer.getvalue(), "test://")


def test_parse_funding_normalises_and_dedupes():
    rows = [
        {"fundingTime": 0, "fundingRate": "0.0001", "markPrice": "20000"},
        {"fundingTime": 0, "fundingRate": "0.0001", "markPrice": "20000"},
        {"fundingTime": 28_800_000, "fundingRate": "-0.0002", "markPrice": "20100"},
    ]
    df = binance_api.parse_funding(rows, "BTCUSDT")
    assert df.height == 2
    assert df["symbol"].unique().to_list() == ["BTCUSDT"]


def test_parse_funding_tolerates_a_missing_mark_price():
    df = binance_api.parse_funding([{"fundingTime": 0, "fundingRate": "0.0001"}], "BTCUSDT")
    assert df["mark_price"].null_count() == 1


def test_empty_funding_payload_gives_a_typed_empty_frame():
    assert binance_api.parse_funding([], "BTCUSDT").height == 0


def test_funding_interval_is_inferred_not_assumed():
    """Some symbols moved to 4h; the cadence comes from the data (§5.1)."""
    eight = pl.DataFrame(
        {
            "funding_time": [0, 28_800_000, 57_600_000],
            "symbol": ["B"] * 3,
            "funding_rate": [0.0] * 3,
            "mark_price": [1.0] * 3,
        }
    )
    four = eight.with_columns(pl.col("funding_time") // 2)
    assert binance_api.infer_funding_interval_hours(eight) == pytest.approx(8.0)
    assert binance_api.infer_funding_interval_hours(four) == pytest.approx(4.0)


def test_inferring_an_interval_from_one_point_raises():
    single = pl.DataFrame(
        {"funding_time": [0], "symbol": ["B"], "funding_rate": [0.0], "mark_price": [1.0]}
    )
    with pytest.raises(ValueError, match="at least two settlements"):
        binance_api.infer_funding_interval_hours(single)


def test_parse_open_interest_normalises():
    rows = [
        {"timestamp": 0, "sumOpenInterest": "100.5", "sumOpenInterestValue": "2000000"},
    ]
    df = binance_api.parse_open_interest(rows, "BTCUSDT")
    assert df["oi_base"][0] == pytest.approx(100.5)


def test_open_interest_retention_is_thirty_days():
    assert binance_api.OI_RETENTION_DAYS == 30


def test_the_oi_notice_names_a_concrete_lost_date():
    notice = open_interest_loss_notice(dt.datetime(2026, 8, 27, tzinfo=dt.UTC))
    assert "2026-07-28" in notice
    assert "PERMANENTLY UNRECOVERABLE" in notice


def test_months_between_is_inclusive():
    got = list(months_between(to_ms("2019-11-01"), to_ms("2020-02-15")))
    assert got == [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]


def test_months_between_a_single_month():
    assert list(months_between(to_ms("2019-01-05"), to_ms("2019-01-25"))) == [(2019, 1)]


def test_quality_reports_are_persisted(tmp_path, bars):
    report = check_ohlcv(bars, "BTCUSDT", "1h")
    paths = write_quality_reports([report], tmp_path)
    stored = json.loads(paths[0].read_text())
    assert stored["symbol"] == "BTCUSDT" and stored["passed"] is True


# ---- archive funding (the path that works without the REST API) ---------------------

FUNDING_CSV = "\n".join(
    ["calc_time,funding_interval_hours,last_funding_rate"]
    + [f"{1_546_300_800_000 + i * 28_800_000 + 2},8,0.0001{i}" for i in range(6)]
)


def _funding_zip(csv: str = FUNDING_CSV) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("BTCUSDT-fundingRate-2019-01.csv", csv)
    return buffer.getvalue()


def test_funding_archive_uri_matches_the_published_layout():
    obj = ArchiveObject.funding("BTCUSDT", 2021, 1)
    assert obj.uri.endswith("fundingRate/BTCUSDT/BTCUSDT-fundingRate-2021-01.zip")


def test_kline_uri_still_keys_on_bar_size():
    assert ArchiveObject("klines", "BTCUSDT", "1h", 2021, 1).uri.endswith("BTCUSDT-1h-2021-01.zip")


def test_parse_funding_archive_produces_the_canonical_schema():
    df = binance_archive.parse_funding(_funding_zip(), "BTCUSDT", "test://")
    assert df.columns == ["funding_time", "symbol", "funding_rate", "mark_price"]
    assert df.height == 6
    assert df["funding_time"].is_sorted()


def test_archive_funding_has_no_mark_price():
    """The archive carries no mark price; basis must join markPriceKlines instead (§5.1)."""
    df = binance_archive.parse_funding(_funding_zip(), "BTCUSDT", "test://")
    assert df["mark_price"].null_count() == df.height


def test_archive_funding_passes_the_quality_gate():
    df = binance_archive.parse_funding(_funding_zip(), "BTCUSDT", "test://")
    assert check_funding(df, "BTCUSDT").passed


def test_archive_states_the_funding_interval():
    """Better than inferring from gaps — §5.1 warns some symbols moved from 8h to 4h."""
    assert binance_archive.funding_interval_hours(_funding_zip(), "test://") == 8.0


def test_a_funding_interval_change_within_a_month_is_refused():
    """A mid-month cadence change is a contract spec change and must not be silently averaged."""
    mixed = FUNDING_CSV.replace(",8,0.00013", ",4,0.00013")
    with pytest.raises(ArchiveError, match="funding interval changes"):
        binance_archive.funding_interval_hours(_funding_zip(mixed), "test://")


def test_archive_funding_dedupes_repeated_settlements():
    doubled = FUNDING_CSV + "\n" + FUNDING_CSV.split("\n", 1)[1]
    df = binance_archive.parse_funding(_funding_zip(doubled), "BTCUSDT", "test://")
    assert df.height == 6


def test_a_corrupt_funding_archive_raises():
    with pytest.raises(ArchiveError, match="not a valid zip"):
        binance_archive.parse_funding(b"junk", "BTCUSDT", "test://")
