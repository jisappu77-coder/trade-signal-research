"""Bulk history from `data.binance.vision` monthly ZIPs (SPEC.md §5.1).

Bulk klines come from the public archive, **never** the REST API — the REST endpoint is rate-limited,
paginated and incomplete for multi-year pulls.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import httpx
import polars as pl

from cryptolab.data import schemas

BASE = "https://data.binance.vision/data/futures/um/monthly"

# The archive's kline CSV column order. It has no header row.
_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
_MARK_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class ArchiveError(RuntimeError):
    """Raised when an archive object is missing or malformed."""


@dataclass(frozen=True, slots=True)
class ArchiveObject:
    """One monthly ZIP in the archive.

    `interval` is empty for datasets that are not bucketed by bar size (fundingRate).
    """

    dataset: str
    symbol: str
    interval: str
    year: int
    month: int

    @staticmethod
    def funding(symbol: str, year: int, month: int) -> ArchiveObject:
        return ArchiveObject("fundingRate", symbol, "", year, month)

    @property
    def name(self) -> str:
        # Klines are keyed by bar size (BTCUSDT-1h-...); datasets without one use the dataset
        # name in that slot (BTCUSDT-fundingRate-...).
        return f"{self.symbol}-{self.interval or self.dataset}-{self.year}-{self.month:02d}.zip"

    @property
    def uri(self) -> str:
        parts = [BASE, self.dataset, self.symbol]
        if self.interval:
            parts.append(self.interval)
        return "/".join([*parts, self.name])


def parse_klines(raw: bytes, source_uri: str, *, dataset: str = "ohlcv") -> pl.DataFrame:
    """Parse a monthly kline ZIP into the canonical schema.

    Binance switched `open_time` from milliseconds to microseconds for some 2025+ archives, so the
    magnitude is inspected rather than assumed.
    """
    csv_bytes = _read_single_csv(raw, source_uri)
    columns = _MARK_COLUMNS if dataset == "mark_price" else _KLINE_COLUMNS
    df = pl.read_csv(
        csv_bytes,
        has_header=_has_header(csv_bytes),
        new_columns=columns,
        schema_overrides={c: pl.Float64 for c in columns},
    )
    df = df.with_columns(
        _normalise_time(pl.col("open_time")).alias("open_time"),
        _normalise_time(pl.col("close_time")).alias("close_time"),
        pl.col("trades").cast(pl.Int64),
    )
    if dataset == "mark_price":
        return schemas.validate(df.select(list(schemas.MARK_PRICE)), "mark_price")
    return schemas.validate(df.select(list(schemas.OHLCV)), "ohlcv")


def _has_header(csv_bytes: bytes) -> bool:
    first = csv_bytes.split(b"\n", 1)[0]
    return b"open_time" in first or b"open" in first.lower().split(b",")[1:2]


def _normalise_time(expr: pl.Expr) -> pl.Expr:
    """Coerce a timestamp column to int64 UTC milliseconds.

    Values above ~1e15 are microseconds (Binance changed units mid-2025); below that, milliseconds.
    """
    return pl.when(expr.abs() > 1e15).then(expr / 1000).otherwise(expr).round(0).cast(pl.Int64)


def _read_single_csv(raw: bytes, source_uri: str) -> bytes:
    """Extract the one CSV an archive object is expected to contain."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ArchiveError(f"{source_uri}: expected exactly one CSV, found {names}")
            return zf.read(names[0])
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"{source_uri}: not a valid zip archive") from exc


def parse_funding(raw: bytes, symbol: str, source_uri: str) -> pl.DataFrame:
    """Parse a monthly fundingRate ZIP into the canonical schema.

    The archive is the only funding source reachable without a REST call, and it carries
    `funding_interval_hours` explicitly — better than inferring the cadence from settlement gaps
    (§5.1 warns that some symbols moved from 8h to 4h). It carries no mark price, so `mark_price`
    is null here; anything needing basis must join mark-price klines (§5.1).
    """
    csv_bytes = _read_single_csv(raw, source_uri)
    columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    df = pl.read_csv(
        csv_bytes,
        has_header=b"calc_time" in csv_bytes.split(b"\n", 1)[0],
        new_columns=columns,
        schema_overrides={c: pl.Float64 for c in columns},
    )
    out = (
        df.select(
            _normalise_time(pl.col("calc_time")).alias("funding_time"),
            pl.lit(symbol, dtype=pl.Utf8).alias("symbol"),
            pl.col("last_funding_rate").alias("funding_rate"),
            pl.lit(None, dtype=pl.Float64).alias("mark_price"),
        )
        .unique(subset="funding_time", keep="first")
        .sort("funding_time")
    )
    return schemas.validate(out, "funding")


def funding_interval_hours(raw: bytes, source_uri: str) -> float:
    """The interval the archive itself declares, rather than one inferred from gaps."""
    csv_bytes = _read_single_csv(raw, source_uri)
    df = pl.read_csv(
        csv_bytes,
        has_header=b"calc_time" in csv_bytes.split(b"\n", 1)[0],
        new_columns=["calc_time", "funding_interval_hours", "last_funding_rate"],
        schema_overrides={"funding_interval_hours": pl.Float64},
    )
    intervals = df["funding_interval_hours"].unique().to_list()
    if len(intervals) != 1:
        raise ArchiveError(
            f"{source_uri}: funding interval changes within the month ({sorted(intervals)}); "
            "this is a contract spec change and must be handled explicitly (SPEC.md §5.1)"
        )
    return float(intervals[0])


async def fetch(client: httpx.AsyncClient, obj: ArchiveObject) -> bytes | None:
    """Download one archive object. Returns None for a 404 (month not published)."""
    response = await client.get(obj.uri, timeout=120.0)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content
