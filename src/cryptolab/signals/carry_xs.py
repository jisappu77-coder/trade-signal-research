"""Cross-sectional carry selection across a whole universe (SPEC.md §8.3, widened).

**What Phase 6 measured, and why this exists.** On BTC and ETH the sleeve was deployed 47% of the
time and captured a median funding rate of about 11% APR, which arithmetic reduces to roughly
3.4% pre-tax on the book. Funding is idiosyncratic per market: when BTC pays nothing, some other
perpetual usually is. Selecting across many markets raises *both* terms — the share of time capital
is working, and the rate it works at — without touching the strategy rule, the cost model, or the
holding period.

**The selection rule.** At each bar the eligible symbols are ranked by the funding rate in force.
Symbols already holding a slot keep it until their own exit threshold is crossed; free slots go to
the highest-paying eligible candidates. Incumbency is deliberate: re-ranking into a marginally
better symbol every bar would pay four fills to chase a few basis points, and the cost model would
be right to punish it.

**Causality.** Everything at bar `t` is computed from data at or before `t`, and the backtest fills
at `t+1`'s open. Ranking uses the rate *in force* — forward-filled from the last settlement — which
is what a trader would actually see on the screen, not the rate that settles next.

**What this does not do.** It does not size positions by conviction, lever the book, or net margin
across symbols. Each slot is an equal, independent, isolated-margin sleeve, which is what a retail
operator running two venues actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# More slots than this and each is too small to clear the venue's minimum order size on a retail
# book; fewer and one symbol's liquidation dominates the result.
DEFAULT_MAX_POSITIONS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class SymbolPanel:
    """One symbol's inputs to the cross-sectional decision, already on the master timeline.

    `funding_apr` is `nan` where the symbol had no data — before it listed, after it was delisted,
    or on a bar either leg was missing. `entry_allowed` is false where a fill would breach §7's
    participation limit, so a market too thin for the slot size excludes itself.
    """

    symbol: str
    funding_apr: np.ndarray
    entry_allowed: np.ndarray

    def __post_init__(self) -> None:
        if self.funding_apr.shape != self.entry_allowed.shape:
            raise ValueError(f"{self.symbol}: funding_apr and entry_allowed must align")


def select(
    panels: list[SymbolPanel],
    *,
    entry_threshold_apr: float,
    exit_threshold_apr: float,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> dict[str, np.ndarray]:
    """Allocate at most `max_positions` slots per bar, returning a 0/1 target per symbol.

    Incumbents are resolved before challengers, so a held slot is given up only when that symbol's
    own exit threshold is crossed, its data ends, or it is liquidated downstream — never merely
    because something better appeared.
    """
    if max_positions < 1:
        raise ValueError("max_positions must be at least 1")
    if not panels:
        return {}
    if exit_threshold_apr > entry_threshold_apr:
        raise ValueError("exit threshold above entry threshold would deploy and unwind every bar")

    length = len(panels[0].funding_apr)
    if any(len(panel.funding_apr) != length for panel in panels):
        raise ValueError("every panel must be on the same master timeline")

    apr = np.vstack([panel.funding_apr for panel in panels])
    allowed = np.vstack([panel.entry_allowed for panel in panels])
    live = np.isfinite(apr)
    # A bar with no data is not a bar with zero funding; ranking must never prefer a dead market.
    rankable = np.where(live, apr, -np.inf)

    targets = np.zeros_like(apr, dtype=float)
    held: set[int] = set()
    for t in range(length):
        # 1. Incumbents: keep the slot unless the exit threshold is crossed or the data ran out.
        held = {i for i in held if live[i, t] and apr[i, t] > exit_threshold_apr}

        # 2. Challengers: eligible, clearing entry, tradeable at this bar's depth, not already in.
        free = max_positions - len(held)
        if free > 0:
            eligible = np.flatnonzero(live[:, t] & allowed[:, t] & (apr[:, t] > entry_threshold_apr))
            candidates = [int(i) for i in eligible if int(i) not in held]
            # Highest funding first; `sorted` is stable, so ties break on the universe's own order
            # rather than on anything that varies between runs.
            candidates.sort(key=lambda i: float(rankable[i, t]), reverse=True)
            held.update(candidates[:free])

        for i in held:
            targets[i, t] = 1.0

    return {panel.symbol: targets[i] for i, panel in enumerate(panels)}
