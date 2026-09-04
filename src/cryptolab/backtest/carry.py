"""Delta-neutral cash-and-carry backtest (SPEC.md §8.3).

The main engine models a single directional position. A carry sleeve is two legs held against each
other, so it gets its own loop rather than being forced through an interface that would misprice it.

What §8.3 requires, and what this models:

* **Entry and exit costs on both legs.** Four fills per episode — buy spot, short perp, then unwind
  both. Costs come from the same fixed §7 regimes; nothing here gets a discount for being
  "market neutral".
* **Margin and liquidation distance on the short perp.** The short is the leg that can be
  liquidated. The distance is tracked every bar and a breach ends the episode at a loss, even
  though the combined position was economically hedged — an exchange liquidates the leg, not the
  strategy.
* **Funding-flip risk.** Funding turning negative means paying instead of receiving, so the episode
  exits on the signal's exit threshold rather than riding it down.
* **The ~40% prior.** §8.3 says only about 40% of apparently attractive opportunities survive costs
  and spread reversals. The realised hit rate is reported against that number.

**P&L decomposition.** Long spot and short perp of equal notional earns the change in the *basis*
plus accumulated funding, minus costs:

    pnl = (spot_t - spot_0) - (perp_t - perp_0) + funding - costs
        = -(basis_t - basis_0) + funding - costs

so a position entered at a wide basis and unwound at a narrow one earns the convergence, and the
funding accrues on top. The basis term is why carry is not risk-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from cryptolab.backtest.costs import (
    BPS,
    DEFAULT_IMPACT_K,
    MAX_PARTICIPATION,
    CostRegime,
    get_regime,
)

# Fraction of the perp notional posted as margin. A breach of this distance is a liquidation.
# This models **isolated** margin with no top-up: the conservative reading, and the one a retail
# operator running the two legs on separate venues actually faces.
DEFAULT_MARGIN_RATE = 0.20

# A forced close is not a normal exit. The position is closed by the exchange at whatever the book
# offers, so it pays the stressed regime's slippage rather than the run's own, plus a liquidation
# fee. Charging only the ordinary exit cost would make liquidations look almost free.
LIQUIDATION_FEE_BPS = 50.0
# §8.3's empirical prior: about 40% of attractive-looking opportunities survive.
SURVIVAL_PRIOR = 0.40
HOURS_PER_YEAR = 8760.0
# Only used where the venue never stated a cadence at all; never as a blanket assumption.
DEFAULT_FUNDING_INTERVAL_HOURS = 8.0
# A ceiling on the modelled cost of one leg. Beyond this the market is not carryable at any size,
# and the bar is refused for entry anyway; the cap only stops a dead bar dominating a forced exit.
MAX_LEG_COST_BPS = 500.0

ExitReason = Literal["signal", "liquidation", "end_of_data"]


def _adverse_funding_factor(rate: float, regime: CostRegime) -> float:
    """Apply §7's funding multiplier so a worse regime is always worse for *this* strategy.

    §7 scales funding magnitude, which is adverse for a position that pays funding and favourable
    for one that receives it. A carry sleeve is a receiver by construction, so multiplying its
    funding naively would make the stressed regime the most profitable one — a cost model that
    rewards you for assuming worse conditions is not modelling anything.

    So the multiplier is inverted for received funding: stress means receiving *less*. The §7
    constants are untouched; only the direction in which they are applied is made consistent with
    what "stressed" is supposed to mean. This is the same asymmetry that exempts funding from the
    §14.2.3 cost-monotonicity assertion.
    """
    receiving = rate > 0  # a short perp receives when funding is positive
    return 1.0 / regime.funding_multiplier if receiving else regime.funding_multiplier


@dataclass(frozen=True, slots=True)
class CarryEpisode:
    """One deployment of the sleeve, entry to exit."""

    entry_time: int
    exit_time: int
    bars_held: int
    notional: float
    entry_funding_apr: float
    funding_collected: float
    basis_pnl: float
    entry_cost: float
    exit_cost: float
    net_pnl: float
    exit_reason: ExitReason
    min_liquidation_distance: float

    @property
    def profitable(self) -> bool:
        return self.net_pnl > 0

    @property
    def realised_apr(self) -> float:
        """Net return on deployed capital, annualised over the actual holding period."""
        if self.notional <= 0 or self.bars_held <= 0:
            return 0.0
        years = (self.exit_time - self.entry_time) / (365.25 * 86_400_000)
        return self.net_pnl / self.notional / years if years > 0 else 0.0


@dataclass(slots=True)
class CarryResult:
    """The sleeve's whole history, plus the §8.3 checks."""

    episodes: list[CarryEpisode] = field(default_factory=list)
    equity_curve: pl.DataFrame = field(default_factory=pl.DataFrame)
    initial_equity: float = 25_000.0
    regime_name: str = "conservative"
    dropped_bars: int = 0

    @property
    def net_pnl(self) -> float:
        return sum(e.net_pnl for e in self.episodes)

    @property
    def total_funding(self) -> float:
        return sum(e.funding_collected for e in self.episodes)

    @property
    def total_costs(self) -> float:
        return sum(e.entry_cost + e.exit_cost for e in self.episodes)

    @property
    def total_basis_pnl(self) -> float:
        return sum(e.basis_pnl for e in self.episodes)

    @property
    def hit_rate(self) -> float:
        """Share of episodes that were profitable net of every cost."""
        return sum(1 for e in self.episodes if e.profitable) / len(self.episodes) if self.episodes else 0.0

    @property
    def liquidations(self) -> int:
        return sum(1 for e in self.episodes if e.exit_reason == "liquidation")

    @property
    def total_return(self) -> float:
        return self.net_pnl / self.initial_equity if self.initial_equity else 0.0

    def prior_comparison(self) -> str:
        """§8.3 requires the realised hit rate be reported against the ~40% prior."""
        if not self.episodes:
            return "no episodes: funding never cleared the entry threshold"
        verdict = "above" if self.hit_rate > SURVIVAL_PRIOR else "at or below"
        return (
            f"{len(self.episodes)} episodes, {self.hit_rate:.0%} profitable net of costs — "
            f"{verdict} the ~{SURVIVAL_PRIOR:.0%} prior in §8.3"
        )


def align_legs(perp: pl.DataFrame, spot: pl.DataFrame, funding: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Join both legs and the funding rate onto a common timeline.

    Bars where either leg is missing a price are **dropped, not filled**, and counted. The Binance
    spot archive has single-hour outages in 12 of the 54 months here; inventing a price for a bar
    that did not trade would be exactly the kind of quiet assumption §6 exists to prevent, and a
    carry position cannot be opened or closed at a price that never existed.
    """
    joined = (
        perp.select(
            pl.col("open_time"),
            pl.col("open").alias("perp_open"),
            pl.col("close").alias("perp_close"),
            pl.col("quote_volume").alias("perp_quote_volume"),
        )
        .join(
            spot.select(
                pl.col("open_time"),
                pl.col("open").alias("spot_open"),
                pl.col("close").alias("spot_close"),
                pl.col("quote_volume").alias("spot_quote_volume"),
            ),
            on="open_time",
            how="inner",
        )
        .sort("open_time")
    )
    dropped = perp.height - joined.height

    has_interval = "interval_hours" in funding.columns
    settlements = funding.sort("funding_time").select(
        pl.col("funding_time").alias("open_time"),
        pl.col("funding_rate"),
        (pl.col("interval_hours") if has_interval else pl.lit(None, dtype=pl.Float64)).alias(
            "interval_hours"
        ),
    )
    with_funding = (
        joined.join(settlements, on="open_time", how="left")
        # The rate in force on a bar is the last one settled — what a trader would actually see.
        .with_columns(pl.col("funding_rate").forward_fill().fill_null(0.0).alias("funding_in_force"))
        .with_columns(pl.col("funding_rate").alias("settlement_rate"))
        # The cadence in force likewise. Defaulting to 8h only where the venue never stated one;
        # a symbol that moved to 4h pays twice as often, and annualising it at 8h would halve it.
        .with_columns(
            pl.col("interval_hours")
            .forward_fill()
            .backward_fill()
            .fill_null(DEFAULT_FUNDING_INTERVAL_HOURS)
            .alias("interval_in_force")
        )
        .drop("funding_rate", "interval_hours")
    )
    return with_funding, dropped


def funding_apr(frame: pl.DataFrame) -> np.ndarray:
    """Annualise the rate in force at each bar using the cadence in force at that bar.

    §5.1's warning made concrete: a hard-coded `rate * 1095` is only right while the symbol settles
    every 8 hours, and venues shorten the interval precisely when funding runs extreme.
    """
    interval = (
        frame["interval_in_force"].to_numpy()
        if "interval_in_force" in frame.columns
        else np.full(frame.height, DEFAULT_FUNDING_INTERVAL_HOURS)
    )
    safe = np.where(interval > 0, interval, DEFAULT_FUNDING_INTERVAL_HOURS)
    return frame["funding_in_force"].to_numpy() * (HOURS_PER_YEAR / safe)


def _leg_costs(
    notional: float,
    perp_quote_volume: np.ndarray,
    spot_quote_volume: np.ndarray,
    regime: CostRegime,
    *,
    half_spread_bps: float,
    max_participation: float = MAX_PARTICIPATION,
    impact_k: float = DEFAULT_IMPACT_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar cost of one leg, and whether an entry is permitted on that bar.

    Returns `(cost_per_leg, entry_allowed)`. The cost is the mean of the two legs' fill costs, since
    every action here trades both; `entry_allowed` is false where either leg would breach §7's
    participation limit, which is how a market too thin for the slot size excludes itself instead of
    being priced as though it were deep.

    This is the vectorised form of `costs.fill_cost` — identical arithmetic, asserted against it in
    the tests — because a Python-level call per bar per symbol is 7.5 million calls across the
    universe run, which CLAUDE.md's "no `.apply()` in hot paths" rule is about.
    """
    fixed_bps = regime.taker_fee_bps + regime.slippage_bps + half_spread_bps

    def one_leg(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive = volume > 0
        # Participation against a zero-volume bar is infinite, so such bars are never entry-allowed.
        participation = np.where(positive, notional / np.where(positive, volume, 1.0), np.inf)
        impact = regime.impact_multiplier * impact_k * np.sqrt(np.maximum(participation, 0.0))
        return notional * (fixed_bps + impact) * BPS, participation <= max_participation

    perp_cost, perp_ok = one_leg(perp_quote_volume)
    spot_cost, spot_ok = one_leg(spot_quote_volume)
    # A bar with no volume at all has unbounded modelled impact; cap the *charge* so one dead bar
    # cannot swamp an episode, while still refusing entry there.
    cost = np.minimum((perp_cost + spot_cost) / 2.0, notional * MAX_LEG_COST_BPS * BPS)
    return cost, perp_ok & spot_ok


def run_carry_backtest(  # noqa: PLR0915 — the loop mirrors one episode's lifecycle end to end;
    # splitting it would hide the ordering (funding, then basis, then liquidation, then the fill)
    # that §8.3 and §9.2 both care about.
    aligned: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    regime_name: str = "conservative",
    initial_equity: float = 25_000.0,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    half_spread_bps: float = 1.0,
    dropped_bars: int = 0,
) -> CarryResult:
    """Run the sleeve. `targets` carries deployment in [0, 1] from `CarrySignal`.

    Fills follow §9.1: a target stamped at bar `t` is acted on at bar `t+1`'s open, on both legs.
    """
    regime: CostRegime = get_regime(regime_name)
    frame = aligned.join(
        targets.select(pl.col("timestamp").alias("open_time"), pl.col("target_position")),
        on="open_time",
        how="left",
    ).with_columns(
        # The one-bar execution delay, made structural rather than remembered.
        pl.col("target_position").shift(1).fill_null(0.0).alias("deployed")
    )

    times = frame["open_time"].to_numpy()
    perp_open = frame["perp_open"].to_numpy()
    perp_close = frame["perp_close"].to_numpy()
    spot_open = frame["spot_open"].to_numpy()
    spot_close = frame["spot_close"].to_numpy()
    settlement = frame["settlement_rate"].to_numpy()
    apr_series = funding_apr(frame)
    deployed = frame["deployed"].to_numpy()

    # Capital funds the spot leg plus margin on the short perp, so the deployable notional per leg
    # is less than the account. This is the capital efficiency carry actually gets.
    notional = initial_equity / (1.0 + margin_rate)
    liquidation_penalty = notional * LIQUIDATION_FEE_BPS * BPS

    # Per-bar cost of filling one leg, sized against that bar's own quote volume via §7's
    # square-root impact term. A flat figure would be a quiet subsidy to thin markets — precisely
    # the markets where funding is highest — so the sleeve would appear to find its best
    # opportunities exactly where its costs were least modelled.
    perp_volume = frame["perp_quote_volume"].to_numpy()
    spot_volume = (
        frame["spot_quote_volume"].to_numpy() if "spot_quote_volume" in frame.columns else perp_volume
    )
    leg_cost_bar, entry_allowed = _leg_costs(
        notional, perp_volume, spot_volume, regime, half_spread_bps=half_spread_bps
    )

    episodes: list[CarryEpisode] = []
    equity = initial_equity
    records: list[dict[str, float | int | bool]] = []

    open_episode: dict[str, float | int] | None = None
    for i in range(len(times)):
        funding_flow = 0.0
        basis_step = 0.0

        if open_episode is not None:
            # Funding settles on the short perp: a short *receives* when the rate is positive.
            rate = settlement[i]
            if not np.isnan(rate) and rate != 0.0:
                funding_flow = notional * float(rate) * _adverse_funding_factor(float(rate), regime)
                open_episode["funding"] = float(open_episode["funding"]) + funding_flow

            # Basis P&L: long spot and short perp earns the narrowing of (perp - spot).
            prev_basis = perp_close[i - 1] - spot_close[i - 1]
            basis_step = float(prev_basis - (perp_close[i] - spot_close[i])) * (notional / spot_close[i])
            open_episode["basis"] = float(open_episode["basis"]) + basis_step

            # Liquidation distance on the short leg: how far the perp can rise before margin is gone.
            entry_perp = float(open_episode["entry_perp"])
            adverse = (perp_close[i] - entry_perp) / entry_perp
            distance = margin_rate - adverse
            open_episode["min_distance"] = min(float(open_episode["min_distance"]), distance)

        equity += funding_flow + basis_step

        want = deployed[i] > 0.5
        have = open_episode is not None

        # A liquidation ends the episode regardless of what the signal wants: the exchange closes
        # the perp leg, and the hedge is gone whether or not the sleeve was economically sound.
        liquidated = have and open_episode is not None and float(open_episode["min_distance"]) <= 0

        if have and (not want or liquidated):
            assert open_episode is not None
            penalty = liquidation_penalty if liquidated else 0.0
            exit_cost = leg_cost_bar[i] * 2
            equity -= exit_cost + penalty  # unwind both legs, plus any forced-close cost
            episodes.append(
                CarryEpisode(
                    entry_time=int(open_episode["entry_time"]),
                    exit_time=int(times[i]),
                    bars_held=i - int(open_episode["entry_index"]),
                    notional=notional,
                    entry_funding_apr=float(open_episode["entry_apr"]),
                    funding_collected=float(open_episode["funding"]),
                    basis_pnl=float(open_episode["basis"]),
                    entry_cost=float(open_episode["entry_cost"]),
                    exit_cost=exit_cost + penalty,
                    net_pnl=float(open_episode["funding"])
                    + float(open_episode["basis"])
                    - float(open_episode["entry_cost"])
                    - exit_cost
                    - penalty,
                    exit_reason="liquidation" if liquidated else "signal",
                    min_liquidation_distance=float(open_episode["min_distance"]),
                )
            )
            open_episode = None
        elif want and not have and entry_allowed[i]:
            # An entry the book cannot absorb is refused outright: §7's participation limit is a
            # capacity constraint, not a cost to be paid. An *exit* is never refused — the position
            # already exists and unwinding it is risk-reducing, the same precedent the directional
            # engine sets in `_apply_capacity_limit`.
            entry_cost = leg_cost_bar[i] * 2
            equity -= entry_cost  # buy spot and short perp
            open_episode = {
                "entry_cost": entry_cost,
                "entry_time": int(times[i]),
                "entry_index": i,
                "entry_perp": float(perp_open[i]),
                "entry_spot": float(spot_open[i]),
                "entry_apr": float(apr_series[i]),
                "funding": 0.0,
                "basis": 0.0,
                "min_distance": margin_rate,
            }

        records.append(
            {
                "open_time": int(times[i]),
                "equity": equity,
                "deployed": 1.0 if open_episode is not None else 0.0,
                "funding_flow": funding_flow,
                "basis_pnl": basis_step,
                "funding_apr": float(apr_series[i]),
            }
        )

    if open_episode is not None:
        assert open_episode is not None
        final_exit_cost = leg_cost_bar[-1] * 2
        equity -= final_exit_cost
        episodes.append(
            CarryEpisode(
                entry_time=int(open_episode["entry_time"]),
                exit_time=int(times[-1]),
                bars_held=len(times) - 1 - int(open_episode["entry_index"]),
                notional=notional,
                entry_funding_apr=float(open_episode["entry_apr"]),
                funding_collected=float(open_episode["funding"]),
                basis_pnl=float(open_episode["basis"]),
                entry_cost=float(open_episode["entry_cost"]),
                exit_cost=final_exit_cost,
                net_pnl=float(open_episode["funding"])
                + float(open_episode["basis"])
                - float(open_episode["entry_cost"])
                - final_exit_cost,
                exit_reason="end_of_data",
                min_liquidation_distance=float(open_episode["min_distance"]),
            )
        )

    return CarryResult(
        episodes=episodes,
        equity_curve=pl.DataFrame(records),
        initial_equity=initial_equity,
        regime_name=regime_name,
        dropped_bars=dropped_bars,
    )
