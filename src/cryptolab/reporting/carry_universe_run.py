"""The Phase 6b universe run: CARRY across every tradeable market, and its verdict.

Phase 6 established that CARRY works on BTC and ETH and earns too little to be worth running: about
3.4% pre-tax, 2.4% post-tax, against a ~7% risk-free deposit. The decomposition of that number is

    book APR  ~=  funding captured  x  capital efficiency  x  deployment fraction  -  costs

and on two symbols the sleeve captured a median of about 11% APR while deployed under half the time.
Both of those terms are properties of the *universe*, not of the strategy rule, which is why this
run changes the universe and nothing else. The entry rule, the hysteresis, the cost regimes, the
margin model and the holding periods are all exactly as Phase 6 left them.

**The search is registered before it is run** (§10.4). Widening a universe is a larger search, and
the honest cost of a larger search is a higher `N` and a correspondingly lower deflated Sharpe for
anything this family ever reports.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptolab.backtest.carry_portfolio import (
    PortfolioResult,
    load_aligned,
    run_portfolio,
    with_capacity,
)
from cryptolab.backtest.costs import get_regime
from cryptolab.data.store import ParquetStore
from cryptolab.signals.carry import DEFAULT_MARGIN_BUFFER, entry_threshold_apr
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.tax import (
    DEFAULT_FD_RATE_APR,
    DEFAULT_SLAB_RATE,
    fixed_deposit_hurdle_apr,
    tax_single_run,
)

STRATEGY_FAMILY = "carry"
SIGNAL = "carry_xs"

# The declared search. Every axis is a decision a person must make before running the sleeve; none
# of them is a knob on the cost model, and none was chosen after seeing a result.
HOLDING_DAYS = (1.0, 3.0, 7.0)
EXIT_FRACTIONS = (0.0, 0.25, 0.5)
MAX_POSITIONS = (4, 8, 12)
MARGIN_RATES = (0.20, 0.50, 1.00)


def declared_grid() -> list[dict[str, Any]]:
    """Every configuration this phase will try, enumerated before any of them is run."""
    return [
        {
            "min_holding_days": holding,
            "exit_fraction": exit_fraction,
            "max_positions": positions,
            "margin_rate": margin,
        }
        for holding, exit_fraction, positions, margin in itertools.product(
            HOLDING_DAYS, EXIT_FRACTIONS, MAX_POSITIONS, MARGIN_RATES
        )
    ]


@dataclass(frozen=True, slots=True)
class UniverseRun:
    """One configuration's outcome, in the terms the verdict is written in."""

    min_holding_days: float
    exit_fraction: float
    max_positions: int
    margin_rate: float
    entry_threshold_apr: float
    episodes: int
    symbols_traded: int
    hit_rate: float
    liquidations: int
    liquidation_rate: float
    deployment_fraction: float
    mean_slots_used: float
    net_pnl: float
    total_funding: float
    total_basis_pnl: float
    total_costs: float
    pre_tax_apr: float
    post_tax_apr: float

    def beats_fixed_deposit(
        self,
        fd_rate_apr: float = DEFAULT_FD_RATE_APR,
        slab_rate: float = DEFAULT_SLAB_RATE,
    ) -> bool:
        """Does this clear the risk-free alternative, **post-tax on both sides**?

        Comparing a 30%-taxed strategy return against an *untaxed* deposit rate was the error this
        method exists to prevent: Indian FD interest is taxed at the holder's slab rate, so a 7%
        deposit is 4.82% post-tax at the top bracket and only 7% for someone below the rebate.
        """
        return self.post_tax_apr > fixed_deposit_hurdle_apr(fd_rate_apr, slab_rate)


def summarise(result: PortfolioResult, params: dict[str, Any], entry: float, capital: float) -> UniverseRun:
    episodes = result.episodes
    pre_tax = result.apr
    traded = sum(e.notional * 4.0 for e in episodes)  # four fills per episode, both legs both ways
    taxed = tax_single_run(
        pre_tax_pnl=result.net_pnl,
        traded_notional=traded,
        initial_equity=capital,
        years=result.span_years,
    )
    post_tax = (
        taxed.post_tax_pnl / capital / result.span_years if capital > 0 and result.span_years > 0 else 0.0
    )
    return UniverseRun(
        min_holding_days=float(params["min_holding_days"]),
        exit_fraction=float(params["exit_fraction"]),
        max_positions=int(params["max_positions"]),
        margin_rate=float(params["margin_rate"]),
        entry_threshold_apr=entry,
        episodes=len(episodes),
        symbols_traded=result.symbols_traded,
        hit_rate=result.hit_rate,
        liquidations=result.liquidations,
        liquidation_rate=result.liquidations / len(episodes) if episodes else 0.0,
        deployment_fraction=result.deployment_fraction,
        mean_slots_used=result.mean_slots_used,
        net_pnl=result.net_pnl,
        total_funding=result.total_funding,
        total_basis_pnl=result.total_basis_pnl,
        total_costs=result.total_costs,
        pre_tax_apr=pre_tax,
        post_tax_apr=post_tax,
    )


def run_universe_grid(
    store: ParquetStore,
    registry: TrialRegistry,
    symbols: list[str],
    *,
    start: str,
    end: str,
    capital: float = 25_000.0,
    regime_name: str = "conservative",
    exchange: str = "binance",
    progress: Any = None,
) -> tuple[list[UniverseRun], int, list[str]]:
    """Register the search, then run it. Returns the runs, the family's `N`, and the symbols used."""
    grid = declared_grid()
    period = f"{start}:{end}"
    # One trial per configuration, not per configuration per symbol: a portfolio run consumes the
    # whole universe at once, so the universe is part of the configuration rather than an axis of it.
    registry.register_grid(
        signal=SIGNAL,
        grid=grid,
        symbols=["UNIVERSE"],
        period=period,
        strategy_family=STRATEGY_FAMILY,
        note=f"phase 6b: cross-sectional carry over {len(symbols)} symbols",
    )
    n = registry.count(strategy_family=STRATEGY_FAMILY)

    round_trip = get_regime(regime_name).round_trip_taker_bps
    runs: list[UniverseRun] = []

    # The lake is read once. Only the capacity mask depends on slot size, and recomputing that is
    # arithmetic on arrays already in memory — caching a copy of every aligned frame per grid cell
    # would be several gigabytes for no gain.
    aligned = [
        loaded
        for symbol in symbols
        if (loaded := load_aligned(store, symbol, start, end, exchange=exchange)) is not None
    ]

    for index, params in enumerate(grid, start=1):
        positions = int(params["max_positions"])
        margin = float(params["margin_rate"])
        loaded_set = [
            with_capacity(
                *entry_data,
                capital=capital,
                max_positions=positions,
                margin_rate=margin,
                regime_name=regime_name,
            )
            for entry_data in aligned
        ]
        entry = entry_threshold_apr(round_trip, float(params["min_holding_days"]), DEFAULT_MARGIN_BUFFER)
        result = run_portfolio(
            loaded_set,
            entry_threshold_apr=entry,
            exit_threshold_apr=entry * float(params["exit_fraction"]),
            capital=capital,
            max_positions=positions,
            margin_rate=margin,
            regime_name=regime_name,
        )
        run = summarise(result, params, entry, capital)
        runs.append(run)
        if progress is not None:
            progress(index, len(grid), run)

    used = sorted(symbol for symbol, _, _ in aligned)
    return runs, n, used


def write_results(runs: list[UniverseRun], n: int, symbols: list[str], out: Path) -> Path:
    """Persist every run, including the losing ones. §12: the failure record is the product."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "trials_n": n,
                "symbols": symbols,
                "runs": [asdict(run) for run in runs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
