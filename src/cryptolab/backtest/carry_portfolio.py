"""Running the carry sleeve across a universe on shared capital (SPEC.md §8.3, widened).

**Why this reuses the single-symbol engine rather than replacing it.** `run_carry_backtest` sizes
every position from `initial_equity` once and never re-sizes it — the notional is fixed for the
run, and equity merely accumulates. That makes per-symbol P&L *additive*: a portfolio of `k` equal,
independent, isolated-margin slots is arithmetically identical to running each symbol on `capital/k`
and summing, provided no more than `k` are ever open at once. `signals.carry_xs.select` guarantees
that, so the tested episode, funding, basis and liquidation logic is used unchanged instead of being
written a second time where the two copies could drift apart.

**No compounding, deliberately.** Slot size is fixed at `capital / max_positions` for the whole run.
A compounding book would report geometric growth off a four-year window, which flatters a strategy
whose returns are this small; the fixed-notional reading is the conservative one and is what Phase 6
reported.

**Survivorship.** The universe is enumerated from the archive (see `data.universe`), so symbols that
were later delisted are present for the months they traded and then simply end. A carry sleeve is
most exposed precisely where markets collapse — funding spikes hardest just before a delisting — so
their inclusion is the point, not an inconvenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import polars as pl

from cryptolab.backtest.carry import (
    DEFAULT_MARGIN_RATE,
    CarryEpisode,
    CarryResult,
    _leg_costs,
    align_legs,
    funding_apr,
    run_carry_backtest,
)
from cryptolab.backtest.costs import get_regime
from cryptolab.data.store import ParquetStore
from cryptolab.signals.carry_xs import DEFAULT_MAX_POSITIONS, SymbolPanel, select

MS_PER_YEAR: Final[float] = 365.25 * 86_400_000.0


@dataclass(frozen=True, slots=True)
class SymbolData:
    """One symbol's aligned legs, plus the bars it can actually be entered on."""

    symbol: str
    aligned: pl.DataFrame
    entry_allowed: np.ndarray
    dropped_bars: int

    @property
    def funding_apr(self) -> np.ndarray:
        """Annualised at the cadence the venue actually settled at on each bar, never assumed 8h."""
        return funding_apr(self.aligned)


@dataclass(slots=True)
class PortfolioResult:
    """The universe run: per-symbol results, plus what the sleeve did as one book."""

    per_symbol: dict[str, CarryResult] = field(default_factory=dict)
    capital: float = 25_000.0
    max_positions: int = DEFAULT_MAX_POSITIONS
    regime_name: str = "conservative"
    symbols_considered: int = 0
    span_years: float = 0.0
    occupancy: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def episodes(self) -> list[CarryEpisode]:
        return [episode for result in self.per_symbol.values() for episode in result.episodes]

    @property
    def net_pnl(self) -> float:
        return sum(result.net_pnl for result in self.per_symbol.values())

    @property
    def total_funding(self) -> float:
        return sum(result.total_funding for result in self.per_symbol.values())

    @property
    def total_costs(self) -> float:
        return sum(result.total_costs for result in self.per_symbol.values())

    @property
    def total_basis_pnl(self) -> float:
        return sum(result.total_basis_pnl for result in self.per_symbol.values())

    @property
    def hit_rate(self) -> float:
        episodes = self.episodes
        return sum(1 for e in episodes if e.profitable) / len(episodes) if episodes else 0.0

    @property
    def liquidations(self) -> int:
        return sum(1 for e in self.episodes if e.exit_reason == "liquidation")

    @property
    def apr(self) -> float:
        """Return on the **whole** book, not on deployed capital — idle slots are counted."""
        if self.capital <= 0 or self.span_years <= 0:
            return 0.0
        return self.net_pnl / self.capital / self.span_years

    @property
    def mean_slots_used(self) -> float:
        """Average number of the `max_positions` slots that were occupied."""
        return float(self.occupancy.mean()) if self.occupancy.size else 0.0

    @property
    def deployment_fraction(self) -> float:
        """Share of the book's capital that was actually working, averaged over the run."""
        return self.mean_slots_used / self.max_positions if self.max_positions else 0.0

    @property
    def symbols_traded(self) -> int:
        return sum(1 for result in self.per_symbol.values() if result.episodes)


def load_aligned(
    store: ParquetStore,
    symbol: str,
    start: str,
    end: str,
    *,
    exchange: str = "binance",
) -> tuple[str, pl.DataFrame, int] | None:
    """Read and align one symbol's three legs. Returns None when any leg is missing or empty.

    A missing leg is not an error to be worked around: two legs cannot be carried against one, and
    the symbol simply is not tradeable over that window. Independent of capital, so a grid over slot
    sizes reads the lake once rather than once per configuration.
    """
    frames = {}
    for dataset in ("ohlcv", "spot_ohlcv", "funding"):
        try:
            frame = store.read(dataset, exchange=exchange, symbol=symbol, start=start, end=end)
        except (FileNotFoundError, ValueError):
            return None
        if frame.height == 0:
            return None
        frames[dataset] = frame

    aligned, dropped = align_legs(frames["ohlcv"], frames["spot_ohlcv"], frames["funding"])
    if aligned.height == 0:
        return None
    return symbol, aligned, dropped


def with_capacity(
    symbol: str,
    aligned: pl.DataFrame,
    dropped: int,
    *,
    capital: float = 25_000.0,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    regime_name: str = "conservative",
    half_spread_bps: float = 1.0,
) -> SymbolData:
    """Mark which bars a slot of this size could actually be filled on (§7 participation limit)."""
    notional = (capital / max_positions) / (1.0 + margin_rate)
    _, allowed = _leg_costs(
        notional,
        aligned["perp_quote_volume"].to_numpy(),
        aligned["spot_quote_volume"].to_numpy(),
        get_regime(regime_name),
        half_spread_bps=half_spread_bps,
    )
    return SymbolData(symbol=symbol, aligned=aligned, entry_allowed=allowed, dropped_bars=dropped)


def load_symbol(
    store: ParquetStore,
    symbol: str,
    start: str,
    end: str,
    *,
    exchange: str = "binance",
    capital: float = 25_000.0,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    regime_name: str = "conservative",
    half_spread_bps: float = 1.0,
) -> SymbolData | None:
    """Read, align and mark capacity in one step. Convenience over the two functions above."""
    loaded = load_aligned(store, symbol, start, end, exchange=exchange)
    if loaded is None:
        return None
    return with_capacity(
        *loaded,
        capital=capital,
        max_positions=max_positions,
        margin_rate=margin_rate,
        regime_name=regime_name,
        half_spread_bps=half_spread_bps,
    )


def build_panels(loaded: list[SymbolData]) -> tuple[np.ndarray, list[SymbolPanel]]:
    """Put every symbol on one master timeline, `nan` where a symbol had no data.

    The union of timestamps rather than the intersection: taking the intersection would silently
    restrict the run to the window the *youngest* symbol covers, which is both a lookahead (today's
    roster imposed on 2020) and a large, quiet loss of data.
    """
    master = np.unique(np.concatenate([data.aligned["open_time"].to_numpy() for data in loaded]))
    panels: list[SymbolPanel] = []
    for data in loaded:
        times = data.aligned["open_time"].to_numpy()
        index = np.searchsorted(master, times)
        apr = np.full(len(master), np.nan)
        allowed = np.zeros(len(master), dtype=bool)
        apr[index] = data.funding_apr
        allowed[index] = data.entry_allowed
        panels.append(SymbolPanel(symbol=data.symbol, funding_apr=apr, entry_allowed=allowed))
    return master, panels


def run_portfolio(
    loaded: list[SymbolData],
    *,
    entry_threshold_apr: float,
    exit_threshold_apr: float,
    capital: float = 25_000.0,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    regime_name: str = "conservative",
    half_spread_bps: float = 1.0,
) -> PortfolioResult:
    """Select across the universe, then run each selected sleeve on its slot of capital."""
    if not loaded:
        return PortfolioResult(capital=capital, max_positions=max_positions, regime_name=regime_name)

    master, panels = build_panels(loaded)
    targets = select(
        panels,
        entry_threshold_apr=entry_threshold_apr,
        exit_threshold_apr=exit_threshold_apr,
        max_positions=max_positions,
    )
    occupancy = np.sum(np.vstack([targets[panel.symbol] for panel in panels]), axis=0)
    slot = capital / max_positions

    per_symbol: dict[str, CarryResult] = {}
    for data in loaded:
        times = data.aligned["open_time"].to_numpy()
        index = np.searchsorted(master, times)
        frame = pl.DataFrame(
            {
                "timestamp": times,
                "target_position": targets[data.symbol][index],
                "confidence": np.ones(len(times)),
            }
        )
        if frame["target_position"].sum() == 0:
            continue
        per_symbol[data.symbol] = run_carry_backtest(
            data.aligned,
            frame,
            regime_name=regime_name,
            initial_equity=slot,
            margin_rate=margin_rate,
            half_spread_bps=half_spread_bps,
            dropped_bars=data.dropped_bars,
        )

    span_years = float(master[-1] - master[0]) / MS_PER_YEAR if len(master) > 1 else 0.0
    return PortfolioResult(
        per_symbol=per_symbol,
        capital=capital,
        max_positions=max_positions,
        regime_name=regime_name,
        symbols_considered=len(loaded),
        span_years=span_years,
        occupancy=occupancy,
    )
