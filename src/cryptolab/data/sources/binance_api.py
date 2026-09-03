"""Binance futures REST for the series the archive does not carry (SPEC.md §5.1).

Two datasets live here, both with hard constraints encoded rather than assumed away:

* **funding rate history** — paginated by `startTime`, 8h cadence for most symbols but 4h for some;
  the interval is derived from the data, never hard-coded.
* **open interest** — Binance serves only ~30 days. It **cannot be backfilled**. The collector must
  run daily from day one; `OI_RETENTION_DAYS` exists so callers can compute what is already lost.
"""

from __future__ import annotations

from typing import Any, Final

import httpx
import polars as pl

from cryptolab.data import schemas

FAPI: Final[str] = "https://fapi.binance.com"
FUNDING_PAGE_LIMIT: Final[int] = 1000
OI_PAGE_LIMIT: Final[int] = 500

# §5.1 — the hard constraint. Open-interest history older than this is gone, permanently.
OI_RETENTION_DAYS: Final[int] = 30


class OpenInterestUnavailableError(RuntimeError):
    """Raised when OI is requested for a window older than the exchange's ~30-day retention."""


async def fetch_funding(client: httpx.AsyncClient, symbol: str, start_ms: int, end_ms: int) -> pl.DataFrame:
    """Fetch funding history for `[start_ms, end_ms]`, paginating forward by `startTime`."""
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        response = await client.get(
            f"{FAPI}/fapi/v1/fundingRate",
            params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": FUNDING_PAGE_LIMIT},
            timeout=60.0,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        last = int(page[-1]["fundingTime"])
        if last <= cursor or len(page) < FUNDING_PAGE_LIMIT:
            cursor = last + 1
            if len(page) < FUNDING_PAGE_LIMIT:
                break
        else:
            cursor = last + 1
    return parse_funding(rows, symbol)


def parse_funding(rows: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
    """Normalise REST funding payloads into the canonical schema."""
    if not rows:
        return schemas.empty("funding")
    df = pl.DataFrame(rows)
    mark = (
        pl.col("markPrice").cast(pl.Float64) if "markPrice" in df.columns else pl.lit(None, dtype=pl.Float64)
    )
    out = (
        df.select(
            pl.col("fundingTime").cast(pl.Int64).alias("funding_time"),
            pl.lit(symbol, dtype=pl.Utf8).alias("symbol"),
            pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
            mark.alias("mark_price"),
        )
        .unique(subset="funding_time", keep="first")
        .sort("funding_time")
    )
    return schemas.validate(out, "funding")


def infer_funding_interval_hours(df: pl.DataFrame) -> float:
    """Derive the funding cadence from the data (8h for most symbols, 4h for some — §5.1)."""
    if df.height < 2:
        raise ValueError("need at least two settlements to infer the funding interval")
    median_ms = float(df["funding_time"].diff().drop_nulls().median())  # type: ignore[arg-type]
    return median_ms / 3_600_000.0


async def fetch_open_interest(
    client: httpx.AsyncClient, symbol: str, period: str = "1h", limit: int = OI_PAGE_LIMIT
) -> pl.DataFrame:
    """Fetch the retrievable open-interest window (~30 days). Cannot be backfilled beyond that."""
    response = await client.get(
        f"{FAPI}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": min(limit, OI_PAGE_LIMIT)},
        timeout=60.0,
    )
    response.raise_for_status()
    return parse_open_interest(response.json(), symbol)


def parse_open_interest(rows: list[dict[str, Any]], symbol: str) -> pl.DataFrame:
    """Normalise REST open-interest payloads into the canonical schema."""
    if not rows:
        return schemas.empty("open_interest")
    out = (
        pl.DataFrame(rows)
        .select(
            pl.col("timestamp").cast(pl.Int64),
            pl.lit(symbol, dtype=pl.Utf8).alias("symbol"),
            pl.col("sumOpenInterest").cast(pl.Float64).alias("oi_base"),
            pl.col("sumOpenInterestValue").cast(pl.Float64).alias("oi_quote"),
        )
        .unique(subset="timestamp", keep="first")
        .sort("timestamp")
    )
    return schemas.validate(out, "open_interest")
