"""Live market data from OKX (SPEC.md §7 names OKX alongside Binance for fee anchors).

**Why OKX and not Binance.** `fapi.binance.com` and `api.binance.com` both return HTTP 451 from
some hosts — Binance geo-blocking the caller, not a proxy fault. OKX serves the same instruments
and is one of the venues §7's cost anchors are drawn from.

**This reads. It does not trade.** No API key, no signing, no authenticated endpoint. Every call
here is a public GET, consistent with §0: v1 places no orders and holds no trade-permissioned keys.

**Venue caveat that matters.** The backtest is Binance data; this is OKX. Funding rates differ
between venues — that difference is itself the `XFUND` Tier-2 signal §2 keeps disabled — so a live
OKX reading is indicative of the strategy's state, not a continuation of the backtested series.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Final

import httpx

BASE: Final[str] = "https://www.okx.com"
TIMEOUT: Final[float] = 20.0

# OKX instrument ids for the two symbols this project trades.
INSTRUMENTS: Final[dict[str, tuple[str, str]]] = {
    "BTCUSDT": ("BTC-USDT", "BTC-USDT-SWAP"),
    "ETHUSDT": ("ETH-USDT", "ETH-USDT-SWAP"),
}


class LiveDataError(RuntimeError):
    """Raised when the venue returns something unusable."""


@dataclass(frozen=True, slots=True)
class LiveQuote:
    """One synchronous read of both legs and the funding rate."""

    symbol: str
    observed_at: int
    spot_price: float
    perp_price: float
    funding_rate: float
    funding_interval_hours: float
    next_funding_time: int
    venue: str = "okx"

    @property
    def funding_apr(self) -> float:
        """Annualised from the interval the venue reports, never an assumed 8h (§5.1)."""
        return self.funding_rate * (8760.0 / self.funding_interval_hours)

    @property
    def basis_bps(self) -> float:
        """Perp against spot. Positive means the perp trades rich, which is the carry entry."""
        if self.spot_price <= 0:
            return 0.0
        return (self.perp_price / self.spot_price - 1.0) * 1e4

    @property
    def next_funding_in(self) -> dt.timedelta:
        return dt.datetime.fromtimestamp(self.next_funding_time / 1000, tz=dt.UTC) - dt.datetime.now(
            tz=dt.UTC
        )


def _payload(response: httpx.Response, what: str) -> dict[str, Any]:
    response.raise_for_status()
    body = response.json()
    if body.get("code") != "0" or not body.get("data"):
        raise LiveDataError(f"{what}: unexpected response {body.get('msg') or body}")
    first: dict[str, Any] = body["data"][0]
    return first


async def fetch_quote(client: httpx.AsyncClient, symbol: str) -> LiveQuote:
    """Read spot, perp and funding for one symbol. Three public GETs, no credentials."""
    if symbol not in INSTRUMENTS:
        raise LiveDataError(f"unknown symbol {symbol!r}; known: {sorted(INSTRUMENTS)}")
    spot_id, swap_id = INSTRUMENTS[symbol]

    spot = _payload(
        await client.get(f"{BASE}/api/v5/market/ticker", params={"instId": spot_id}, timeout=TIMEOUT),
        "spot ticker",
    )
    perp = _payload(
        await client.get(f"{BASE}/api/v5/market/ticker", params={"instId": swap_id}, timeout=TIMEOUT),
        "perp ticker",
    )
    funding = _payload(
        await client.get(f"{BASE}/api/v5/public/funding-rate", params={"instId": swap_id}, timeout=TIMEOUT),
        "funding rate",
    )

    this_settlement = int(funding["fundingTime"])
    next_settlement = int(funding.get("nextFundingTime") or 0) or this_settlement
    interval_hours = (
        (next_settlement - this_settlement) / 3_600_000 if next_settlement > this_settlement else 8.0
    )
    return LiveQuote(
        symbol=symbol,
        observed_at=int(perp["ts"]),
        spot_price=float(spot["last"]),
        perp_price=float(perp["last"]),
        funding_rate=float(funding["fundingRate"]),
        funding_interval_hours=interval_hours,
        next_funding_time=next_settlement,
    )


async def fetch_all(client: httpx.AsyncClient, symbols: list[str]) -> list[LiveQuote]:
    return [await fetch_quote(client, symbol) for symbol in symbols]
