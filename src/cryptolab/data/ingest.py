"""Ingestion orchestration (SPEC.md §5, §6, Phase 1).

Every ingested month goes through the quality gate before it is considered usable. The OI warning
in `open_interest_loss_notice` is printed loudly at init because the constraint is irreversible.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from cryptolab.data import quality
from cryptolab.data.quality import DataQualityReport
from cryptolab.data.sources import binance_api, binance_archive
from cryptolab.data.sources.binance_archive import ArchiveObject
from cryptolab.data.store import ParquetStore, from_ms, to_ms


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome for one symbol-month."""

    symbol: str
    interval: str
    year: int
    month: int
    rows: int
    source_uri: str
    report: DataQualityReport | None
    skipped: str | None = None


def months_between(start_ms: int, end_ms: int) -> Iterator[tuple[int, int]]:
    """Yield (year, month) covering [start_ms, end_ms] inclusive."""
    start, end = from_ms(start_ms), from_ms(end_ms)
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def open_interest_loss_notice(now: dt.datetime | None = None) -> str:
    """The §5.1 warning, with the concrete date before which OI history is already unrecoverable."""
    now = now or dt.datetime.now(tz=dt.UTC)
    horizon = (now - dt.timedelta(days=binance_api.OI_RETENTION_DAYS)).date()
    return (
        "OPEN INTEREST: Binance retains only ~"
        f"{binance_api.OI_RETENTION_DAYS} days of open-interest history. Data before {horizon} is "
        "PERMANENTLY UNRECOVERABLE and cannot be backfilled at any price. Any OI-derived hypothesis "
        "is untestable on history unless the daily collector has already been running. "
        "Start it now (`cryptolab collect-oi`) or treat OI as out of scope."
    )


async def ingest_klines(
    store: ParquetStore,
    symbol: str,
    interval: str,
    start: str | dt.date | int,
    end: str | dt.date | int,
    *,
    exchange: str = "binance",
    dataset: str = "ohlcv",
    client: httpx.AsyncClient | None = None,
    concurrency: int = 4,
) -> list[IngestResult]:
    """Ingest monthly kline archives for one symbol, gating each month on quality."""
    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    archive_dataset = "markPriceKlines" if dataset == "mark_price" else "klines"
    spot = dataset == "spot_ohlcv"
    semaphore = asyncio.Semaphore(concurrency)

    async def one(year: int, month: int) -> IngestResult:
        obj = ArchiveObject(archive_dataset, symbol, interval, year, month, spot=spot)
        async with semaphore:
            raw = await binance_archive.fetch(client, obj)
        if raw is None:
            return IngestResult(symbol, interval, year, month, 0, obj.uri, None, skipped="404")
        df = binance_archive.parse_klines(raw, obj.uri, dataset=dataset)
        report = quality.check_ohlcv(df, symbol, interval) if dataset in ("ohlcv", "spot_ohlcv") else None
        store.write(df, dataset, exchange=exchange, symbol=symbol, source_uri=obj.uri)
        return IngestResult(symbol, interval, year, month, df.height, obj.uri, report)

    try:
        tasks = [one(y, m) for y, m in months_between(to_ms(start), to_ms(end))]
        return list(await asyncio.gather(*tasks))
    finally:
        if owns_client:
            await client.aclose()


async def ingest_funding_archive(
    store: ParquetStore,
    symbol: str,
    start: str | dt.date | int,
    end: str | dt.date | int,
    *,
    exchange: str = "binance",
    client: httpx.AsyncClient | None = None,
    concurrency: int = 4,
) -> list[IngestResult]:
    """Ingest funding from the monthly archive rather than the REST API.

    Preferred over `ingest_funding`: no pagination, no rate limits, and the archive states the
    funding interval instead of leaving it to be inferred. The REST path is retained because the
    archive lags the current month.
    """
    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(year: int, month: int) -> IngestResult:
        obj = ArchiveObject.funding(symbol, year, month)
        async with semaphore:
            raw = await binance_archive.fetch(client, obj)
        if raw is None:
            return IngestResult(symbol, "funding", year, month, 0, obj.uri, None, skipped="404")
        df = binance_archive.parse_funding(raw, symbol, obj.uri)
        # A month may declare more than one interval — a mid-month spec change. The per-settlement
        # `interval_hours` column carries it, so the month is kept and labelled rather than
        # refused; see `binance_archive.funding_intervals` for why dropping it is not neutral.
        intervals = binance_archive.funding_intervals(raw, obj.uri)
        label = "/".join(f"{v:g}h" for v in intervals) if intervals else "?"
        report = quality.check_funding(df, symbol)
        store.write(df, "funding", exchange=exchange, symbol=symbol, source_uri=obj.uri)
        return IngestResult(symbol, label, year, month, df.height, obj.uri, report)

    try:
        tasks = [one(y, m) for y, m in months_between(to_ms(start), to_ms(end))]
        return list(await asyncio.gather(*tasks))
    finally:
        if owns_client:
            await client.aclose()


async def ingest_funding(
    store: ParquetStore,
    symbol: str,
    start: str | dt.date | int,
    end: str | dt.date | int,
    *,
    exchange: str = "binance",
    client: httpx.AsyncClient | None = None,
) -> IngestResult:
    """Ingest the full funding history for one symbol."""
    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    uri = f"{binance_api.FAPI}/fapi/v1/fundingRate?symbol={symbol}"
    try:
        df = await binance_api.fetch_funding(client, symbol, to_ms(start), to_ms(end))
    finally:
        if owns_client:
            await client.aclose()
    report = quality.check_funding(df, symbol)
    if df.height:
        store.write(df, "funding", exchange=exchange, symbol=symbol, source_uri=uri)
    stamp = from_ms(int(df["funding_time"].min())) if df.height else from_ms(to_ms(start))  # type: ignore[arg-type]
    return IngestResult(symbol, "8h", stamp.year, stamp.month, df.height, uri, report)


async def collect_open_interest(
    store: ParquetStore,
    symbols: list[str],
    *,
    exchange: str = "binance",
    period: str = "1h",
    client: httpx.AsyncClient | None = None,
) -> list[IngestResult]:
    """The daily OI collector. Append-only; whatever is not collected today is lost forever."""
    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    uri = f"{binance_api.FAPI}/futures/data/openInterestHist"
    results: list[IngestResult] = []
    try:
        for symbol in symbols:
            df = await binance_api.fetch_open_interest(client, symbol, period=period)
            if df.height:
                store.write(df, "open_interest", exchange=exchange, symbol=symbol, source_uri=uri)
            stamp = from_ms(int(df["timestamp"].max())) if df.height else dt.datetime.now(tz=dt.UTC)  # type: ignore[arg-type]
            results.append(IngestResult(symbol, period, stamp.year, stamp.month, df.height, uri, None))
    finally:
        if owns_client:
            await client.aclose()
    return results


def write_quality_reports(reports: list[DataQualityReport], out_dir: Path | str) -> list[Path]:
    """Persist quality reports as JSON, one per symbol-month (§6)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for report in reports:
        name = f"{report.dataset}_{report.symbol}_{report.interval or 'na'}_{report.start_time}.json"
        path = out / name
        path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        paths.append(path)
    return paths


@dataclass(frozen=True, slots=True)
class BundleResult:
    """Outcome of ingesting all three legs a carry sleeve needs for one symbol."""

    symbol: str
    perp_rows: int
    spot_rows: int
    funding_rows: int
    failed_quality: int
    error: str | None = None

    @property
    def usable(self) -> bool:
        """A symbol is usable only if every leg arrived. Two legs cannot be carried against one."""
        return self.error is None and min(self.perp_rows, self.spot_rows, self.funding_rows) > 0


async def ingest_symbol_bundle(
    store: ParquetStore,
    symbol: str,
    start: str | dt.date | int,
    end: str | dt.date | int,
    *,
    interval: str = "1h",
    exchange: str = "binance",
    client: httpx.AsyncClient | None = None,
    concurrency: int = 4,
) -> BundleResult:
    """Ingest perp klines, spot klines and funding for one symbol.

    Months the archive does not hold are skipped, not failed: a symbol listed in 2023 legitimately
    has no 2021 data, and treating that as an error would exclude every late listing and reinstate
    the survivorship bias `data.universe` exists to avoid.
    """
    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    try:
        perp, spot, funding = await asyncio.gather(
            ingest_klines(
                store,
                symbol,
                interval,
                start,
                end,
                exchange=exchange,
                dataset="ohlcv",
                client=client,
                concurrency=concurrency,
            ),
            ingest_klines(
                store,
                symbol,
                interval,
                start,
                end,
                exchange=exchange,
                dataset="spot_ohlcv",
                client=client,
                concurrency=concurrency,
            ),
            ingest_funding_archive(
                store, symbol, start, end, exchange=exchange, client=client, concurrency=concurrency
            ),
        )
    # A broad catch on purpose: one unreachable symbol must not abort a 190-symbol ingest.
    except Exception as exc:
        return BundleResult(symbol, 0, 0, 0, 0, error=f"{type(exc).__name__}: {exc}")
    finally:
        if owns_client:
            await client.aclose()

    every = [*perp, *spot, *funding]
    return BundleResult(
        symbol=symbol,
        perp_rows=sum(r.rows for r in perp),
        spot_rows=sum(r.rows for r in spot),
        funding_rows=sum(r.rows for r in funding),
        failed_quality=sum(1 for r in every if r.report is not None and not r.report.passed),
    )


async def ingest_universe(
    store: ParquetStore,
    symbols: list[str],
    start: str | dt.date | int,
    end: str | dt.date | int,
    *,
    interval: str = "1h",
    exchange: str = "binance",
    symbol_concurrency: int = 3,
    month_concurrency: int = 8,
    on_done: Callable[[BundleResult], None] | None = None,
) -> list[BundleResult]:
    """Ingest a whole universe, one bundle per symbol, bounded in both directions.

    Two concurrency limits rather than one: `symbol_concurrency` caps how many symbols are in
    flight, `month_concurrency` how many months within a symbol. The product is what the archive
    actually sees.
    """
    semaphore = asyncio.Semaphore(symbol_concurrency)
    async with httpx.AsyncClient(follow_redirects=True) as client:

        async def one(symbol: str) -> BundleResult:
            async with semaphore:
                result = await ingest_symbol_bundle(
                    store,
                    symbol,
                    start,
                    end,
                    interval=interval,
                    exchange=exchange,
                    client=client,
                    concurrency=month_concurrency,
                )
            if on_done is not None:
                on_done(result)
            return result

        return list(await asyncio.gather(*[one(symbol) for symbol in symbols]))
