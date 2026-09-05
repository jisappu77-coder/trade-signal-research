"""Discovering the tradeable universe from the archive itself, rather than from memory.

**Why this module exists.** Phase 6 tested CARRY on two symbols and found it profitable but too
small to pay for its risks — the sleeve was deployed under half the time and captured a median
funding rate of about 11% APR. Both terms improve with a wider universe, because funding is
idiosyncratic per market. Widening it honestly is the whole difficulty.

**The survivorship trap, and why the archive escapes it.** Naming the symbols a person remembers
is a performance filter wearing a liquidity costume: the markets that come to mind are the ones
that survived. A funding carry is *precisely* the trade that dies in a collapse — funding spikes
hardest just before a market is delisted — so excluding LUNA, FTT and their kind would bias the
result upward by removing the losses the strategy is most exposed to. `data.binance.vision` keeps
delisted symbols, so the universe here is enumerated from the archive's own key listing and
includes markets that no longer exist.

**Point-in-time membership.** Enumeration alone still describes today. `coverage_months` records
which months each symbol actually has data for, so a backtest can ask what was tradeable at time
`t` instead of assuming today's roster held in 2021.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Final

import httpx

from cryptolab.data.sources.binance_archive import ArchiveObject

# The archive is a public S3 bucket; its list endpoint is what makes enumeration possible.
LISTING_ENDPOINT: Final[str] = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PERP_PREFIX: Final[str] = "data/futures/um/monthly/klines/"
SPOT_PREFIX: Final[str] = "data/spot/monthly/klines/"
LISTING_TIMEOUT: Final[float] = 30.0
PROBE_TIMEOUT: Final[float] = 25.0
PROBE_ATTEMPTS: Final[int] = 3


class UniverseError(RuntimeError):
    """Raised when the archive listing cannot be read or parsed."""


@dataclass(frozen=True, slots=True)
class SymbolCoverage:
    """Which probe months a symbol has a complete perp + spot + funding set for."""

    symbol: str
    months: tuple[tuple[int, int], ...]

    @property
    def listed(self) -> bool:
        return bool(self.months)

    @property
    def first_month(self) -> tuple[int, int] | None:
        return min(self.months) if self.months else None


async def list_archive_symbols(client: httpx.AsyncClient, prefix: str) -> list[str]:
    """Page through the bucket listing and return every symbol directory under `prefix`."""
    pattern = re.compile(rf"<Prefix>{re.escape(prefix)}([^/<]+)/</Prefix>")
    found: list[str] = []
    marker = ""
    while True:
        response = await client.get(
            LISTING_ENDPOINT,
            params={"delimiter": "/", "prefix": prefix, "marker": marker},
            timeout=LISTING_TIMEOUT,
        )
        response.raise_for_status()
        body = response.text
        if "<ListBucketResult" not in body:
            raise UniverseError(f"archive listing for {prefix!r} returned no bucket result")
        found.extend(pattern.findall(body))
        if "<IsTruncated>true" not in body:
            return found
        next_marker = re.search(r"<NextMarker>([^<]+)</NextMarker>", body)
        if next_marker is None:  # pragma: no cover - the bucket always supplies one when truncated
            raise UniverseError(f"truncated listing for {prefix!r} carried no NextMarker")
        marker = next_marker.group(1)


async def discover_symbols(client: httpx.AsyncClient, *, quote: str = "USDT") -> list[str]:
    """Every `quote`-denominated symbol that has **both** a perp and a spot archive.

    A carry sleeve needs both legs, so a perp without a matching spot pair is not tradeable here
    however liquid it is.
    """
    perp, spot = await asyncio.gather(
        list_archive_symbols(client, PERP_PREFIX),
        list_archive_symbols(client, SPOT_PREFIX),
    )
    both = set(perp) & set(spot)
    return sorted(symbol for symbol in both if symbol.endswith(quote))


async def _exists(client: httpx.AsyncClient, obj: ArchiveObject) -> bool:
    """HEAD one archive object, retrying transient transport failures.

    A network blip must not be read as "this symbol did not exist", which would silently shrink the
    universe and reintroduce exactly the bias this module is built to avoid.
    """
    for attempt in range(PROBE_ATTEMPTS):
        try:
            response = await client.head(obj.uri, timeout=PROBE_TIMEOUT)
        except httpx.HTTPError:
            if attempt == PROBE_ATTEMPTS - 1:
                return False
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        return response.status_code == 200
    return False  # pragma: no cover - the loop always returns first


async def probe_coverage(
    client: httpx.AsyncClient,
    symbols: list[str],
    months: list[tuple[int, int]],
    *,
    interval: str = "1h",
    concurrency: int = 24,
) -> list[SymbolCoverage]:
    """For each symbol, which of `months` have perp klines, spot klines **and** funding.

    All three are required: a month missing any leg cannot be backtested as a carry, and filling
    the gap would be the sort of quiet assumption §6 exists to prevent.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def month_complete(symbol: str, year: int, month: int) -> bool:
        async with semaphore:
            legs = await asyncio.gather(
                _exists(client, ArchiveObject("klines", symbol, interval, year, month)),
                _exists(client, ArchiveObject("klines", symbol, interval, year, month, spot=True)),
                _exists(client, ArchiveObject.funding(symbol, year, month)),
            )
        return all(legs)

    async def one(symbol: str) -> SymbolCoverage:
        flags = await asyncio.gather(*[month_complete(symbol, y, m) for y, m in months])
        present = tuple(month for month, ok in zip(months, flags, strict=True) if ok)
        return SymbolCoverage(symbol=symbol, months=present)

    return list(await asyncio.gather(*[one(symbol) for symbol in symbols]))


def tradeable(coverage: list[SymbolCoverage]) -> list[str]:
    """Symbols with a complete set in at least one probe month, sorted for a stable universe id."""
    return sorted(entry.symbol for entry in coverage if entry.listed)
