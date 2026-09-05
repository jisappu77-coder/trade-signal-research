"""Live signal readout — a research instrument's dashboard, not an execution path.

This evaluates the CARRY entry condition against the market right now and says whether the sleeve
would be deployed. It reads public data and prints. It places no orders, holds no keys, and signs
no requests, which is what §0 and §18 require of v1.

A reading is not a recommendation. Nothing here has passed the §11 gates, and CARRY's own backtest
returns roughly 2-3% APR after Indian VDA tax — below a risk-free deposit. The readout exists so
the signal's live state is observable, not so it can be acted on blind.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cryptolab.backtest.costs import get_regime
from cryptolab.data.sources.okx_live import LiveQuote
from cryptolab.signals.carry import DEFAULT_MARGIN_BUFFER, entry_threshold_apr


@dataclass(frozen=True, slots=True)
class LiveSignal:
    """The CARRY condition evaluated against one live quote."""

    symbol: str
    observed_at: int
    venue: str
    spot_price: float
    perp_price: float
    basis_bps: float
    funding_rate: float
    funding_interval_hours: float
    funding_apr: float
    entry_threshold_apr: float
    exit_threshold_apr: float
    holding_days: float
    regime: str
    fires: bool
    headroom_apr: float
    next_funding_in_minutes: float

    @property
    def action(self) -> str:
        if self.fires:
            return "DEPLOY — long spot, short perp (equal notional)"
        return "STAND DOWN — funding does not clear the entry threshold"

    def line(self) -> str:
        """One-line human readout."""
        mark = "SIGNAL" if self.fires else "  none"
        return (
            f"{mark}  {self.symbol:<8} funding {self.funding_apr:+7.2%} APR   "
            f"entry {self.entry_threshold_apr:6.2%}   headroom {self.headroom_apr:+7.2%}   "
            f"basis {self.basis_bps:+6.1f}bps   next settle in "
            f"{self.next_funding_in_minutes:.0f}m"
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "action": self.action}


def evaluate(
    quote: LiveQuote,
    *,
    holding_days: float = 7.0,
    exit_fraction: float = 0.25,
    regime_name: str = "conservative",
    margin_buffer: float = DEFAULT_MARGIN_BUFFER,
) -> LiveSignal:
    """Apply the §8.3 entry rule to a live quote.

    Defaults match the backtest's best-performing configuration (7-day hold), so the live readout
    and the backtested result are the same rule rather than two different ones.
    """
    regime = get_regime(regime_name)
    entry = entry_threshold_apr(regime.round_trip_taker_bps, holding_days, margin_buffer)
    apr = quote.funding_apr
    return LiveSignal(
        symbol=quote.symbol,
        observed_at=quote.observed_at,
        venue=quote.venue,
        spot_price=quote.spot_price,
        perp_price=quote.perp_price,
        basis_bps=quote.basis_bps,
        funding_rate=quote.funding_rate,
        funding_interval_hours=quote.funding_interval_hours,
        funding_apr=apr,
        entry_threshold_apr=entry,
        exit_threshold_apr=entry * exit_fraction,
        holding_days=holding_days,
        regime=regime_name,
        fires=apr > entry,
        headroom_apr=apr - entry,
        next_funding_in_minutes=quote.next_funding_in.total_seconds() / 60.0,
    )


def append_observation(signal: LiveSignal, path: Path | str) -> Path:
    """Append one reading to a JSONL log, so a watch can be reconstructed afterwards."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(signal.to_dict(), default=str) + "\n")
    return target


def read_observations(path: Path | str) -> list[dict[str, object]]:
    """Read back a watch log. Missing file means no observations, not an error."""
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]


def summarise_watch(observations: list[dict[str, object]]) -> str:
    """Summarise a completed watch: how often the signal fired, and how close it came."""
    if not observations:
        return "no observations recorded"
    fired = [o for o in observations if o.get("fires")]
    by_symbol: dict[str, list[dict[str, object]]] = {}
    for observation in observations:
        by_symbol.setdefault(str(observation["symbol"]), []).append(observation)

    lines = [
        f"{len(observations)} observations across {len(by_symbol)} symbols; "
        f"{len(fired)} fired ({len(fired) / len(observations):.0%})"
    ]
    for symbol, rows in sorted(by_symbol.items()):
        aprs = [float(r["funding_apr"]) for r in rows]  # type: ignore[arg-type]
        headroom = [float(r["headroom_apr"]) for r in rows]  # type: ignore[arg-type]
        hits = sum(1 for r in rows if r.get("fires"))
        lines.append(
            f"  {symbol}: {hits}/{len(rows)} fired · funding APR "
            f"{min(aprs):+.2%} to {max(aprs):+.2%} · closest approach "
            f"{max(headroom):+.2%}"
        )
    return "\n".join(lines)


def utc_now_ms() -> int:
    return int(dt.datetime.now(tz=dt.UTC).timestamp() * 1000)
