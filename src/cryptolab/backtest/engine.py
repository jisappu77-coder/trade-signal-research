"""The backtest event loop (SPEC.md §9).

**Bar-close discipline (§9.1)**: signals are computed from bar `t` close, orders are submitted at
bar `t` close, and they fill at bar `t+1` **open** with slippage. There is no intrabar fill logic,
and adding one without tick data is on the §18 refusal list — inner-bar fills on 5m bars are the
single most common source of inflated crypto backtests.

The event loop order is fixed by §9.2 and the implementation follows it literally.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from cryptolab.backtest.costs import (
    BPS,
    CapacityBreachError,
    CostRegime,
    cost_drag_bps_per_year,
    get_regime,
)
from cryptolab.backtest.portfolio import (
    DEFAULT_NO_TRADE_BAND,
    PortfolioState,
    apply_fill,
    apply_funding,
    no_trade_band_filter,
)
from cryptolab.backtest.risk import RiskEngine

MS_PER_YEAR = 365.25 * 86_400_000


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything the loop needs that is not data. Costs are a named regime, never loose numbers."""

    regime_name: str = "conservative"
    initial_equity: float = 100_000.0
    no_trade_band: float = DEFAULT_NO_TRADE_BAND
    max_leverage: float = 2.0
    half_spread_bps: float = 1.0
    impact_k: float = 10.0
    max_participation: float = 0.01
    maker: bool = False
    warmup_bars: int = 0
    # §7: "Reject any fill where order_notional > 0.01 * bar_quote_volume and log it as a capacity
    # breach." Rejecting the fill and counting it is therefore the default; "raise" is the stricter
    # mode used where a breach should stop the run outright.
    on_capacity_breach: Literal["reject", "raise"] = "reject"

    @property
    def regime(self) -> CostRegime:
        return get_regime(self.regime_name)


@dataclass(slots=True)
class BacktestResult:
    """Per-bar record plus the derived metrics every report header needs (§12)."""

    equity_curve: pl.DataFrame
    config: BacktestConfig
    gross_pnl: float
    fees_and_slippage: float
    funding_paid: float
    net_pnl: float
    turnover_per_year: float
    capacity_breaches: int
    partial_fills: int
    bars: int
    interval_ms: int
    kill_reason: str | None = None

    @property
    def returns(self) -> np.ndarray:
        """Per-bar net returns — the series every reported Sharpe is computed from."""
        return self._returns("equity")

    @property
    def gross_returns(self) -> np.ndarray:
        """Per-bar returns before fees, slippage and funding.

        Used by the §14.2.2 shuffle test, which is asking whether a signal has *foresight*. Run on
        net returns it would instead measure the cost drag, and would "pass" any strategy that
        merely trades enough to lose money reliably.
        """
        return self._returns("equity_gross")

    def _returns(self, column: str) -> np.ndarray:
        equity = self.equity_curve[column].to_numpy()
        if equity.size < 2:
            return np.zeros(0, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.diff(equity) / equity[:-1]
        cleaned: np.ndarray = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned

    @property
    def periods_per_year(self) -> float:
        return MS_PER_YEAR / self.interval_ms

    def sharpe(self, *, annualised: bool = False, gross: bool = False) -> float:
        """Per-observation Sharpe by default.

        Annualisation happens only at the reporting boundary (§10.3): the DSR machinery consumes
        per-observation Sharpes, and mixing the two units silently inflates the deflated Sharpe.
        """
        r = self.gross_returns if gross else self.returns
        if r.size < 2:
            return 0.0
        sd = float(np.std(r, ddof=1))
        if sd == 0:
            return 0.0
        sr = float(np.mean(r)) / sd
        return sr * float(np.sqrt(self.periods_per_year)) if annualised else sr

    @property
    def total_return(self) -> float:
        """Net return over the whole run. Monotone in cost regime, unlike absolute P&L.

        Absolute P&L is not comparable across regimes: sizing is equity-proportional, so a
        costlier run trades from a smaller base and can post a *smaller* absolute loss.
        """
        equity = self.equity_curve["equity"].to_numpy()
        if equity.size == 0:
            return 0.0
        return float(equity[-1]) / self.config.initial_equity - 1.0

    @property
    def went_bankrupt(self) -> bool:
        return bool(self.equity_curve["bankrupt"].any())

    @property
    def flat_fraction(self) -> float:
        """Share of bars holding no position.

        A run that spends most of its life flat has metrics computed largely over cash. Sharpe and
        return over such a window describe a short burst of trading padded with nothing, so this is
        reported beside them rather than left for the reader to infer from the equity curve.
        """
        if self.equity_curve.height == 0:
            return 0.0
        flat = (self.equity_curve["position_units"] == 0).sum()
        return float(flat) / self.equity_curve.height

    @property
    def killed_fraction(self) -> float:
        """Share of bars after the risk engine forced a flatten."""
        if self.equity_curve.height == 0:
            return 0.0
        return float(self.equity_curve["killed"].sum()) / self.equity_curve.height

    @property
    def first_kill_time(self) -> int | None:
        """Timestamp of the first kill-switch trip, if any."""
        killed = self.equity_curve.filter(pl.col("killed"))
        return int(killed["open_time"][0]) if killed.height else None

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown as a positive fraction."""
        equity = self.equity_curve["equity"].to_numpy()
        if equity.size == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        return float(np.max((peak - equity) / peak))

    def breakeven_cost_bps(self) -> float:
        """§7.1 — the round-trip cost at which net Sharpe crosses zero.

        Solved directly: net P&L is gross minus (turnover x cost), so the break-even round-trip is
        the gross edge divided by the traded notional. Reported in every header because it is more
        informative than the Sharpe itself.
        """
        traded = float(self.equity_curve["traded_notional"].sum())
        if traded <= 0:
            return 0.0
        edge = self.gross_pnl - self.funding_paid
        return 0.0 if edge <= 0 else edge / traded / BPS

    def cost_drag_bps_per_year(self) -> float:
        return cost_drag_bps_per_year(self.turnover_per_year, self.config.regime.round_trip_taker_bps)

    def summary(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self.config).items()},
            "bars": self.bars,
            "gross_pnl": self.gross_pnl,
            "fees_and_slippage": self.fees_and_slippage,
            "funding_paid": self.funding_paid,
            "net_pnl": self.net_pnl,
            "sharpe_per_bar": self.sharpe(),
            "sharpe_per_bar_gross": self.sharpe(gross=True),
            "total_return": self.total_return,
            "sharpe_annualised": self.sharpe(annualised=True),
            "max_drawdown": self.max_drawdown(),
            "turnover_per_year": self.turnover_per_year,
            "breakeven_cost_bps": self.breakeven_cost_bps(),
            "cost_drag_bps_per_year": self.cost_drag_bps_per_year(),
            "capacity_breaches": self.capacity_breaches,
            "partial_fills": self.partial_fills,
            "went_bankrupt": self.went_bankrupt,
            "flat_fraction": self.flat_fraction,
            "killed_fraction": self.killed_fraction,
            "kill_reason": self.kill_reason,
        }


def _apply_capacity_limit(
    delta_units: float,
    current_units: float,
    fill_price: float,
    bar_volume: float,
    config: BacktestConfig,
) -> tuple[float, int, int]:
    """Apply the §7 participation limit to one order.

    Returns (units_to_trade, breaches, partial_fills). A risk-*increasing* order above the limit
    simply does not happen, per §7. A risk-*reducing* order is instead worked down to the limit and
    continued on later bars: rejecting it outright traps the book, which then cannot delever while
    every corrective trade is refused, and realised leverage runs away. The partial still pays full
    impact, so this is not a discount.
    """
    if fill_price <= 0:
        return 0.0, 0, 0
    cap_units = config.max_participation * bar_volume / fill_price
    if abs(delta_units) <= cap_units:
        return delta_units, 0, 0
    if config.on_capacity_breach == "raise":
        raise CapacityBreachError(
            f"order of {abs(delta_units) * fill_price:,.0f} exceeds the "
            f"{config.max_participation:.2%} limit on a bar of {bar_volume:,.0f}"
        )
    reduces_risk = abs(current_units + delta_units) < abs(current_units)
    if reduces_risk:
        # Shaved by an ulp-ish margin: clipping to exactly the cap lands on the boundary, and
        # `fill_cost` recomputes participation from the rounded notional and rejects it.
        return math.copysign(cap_units * (1.0 - 1e-9), delta_units), 1, 1
    return 0.0, 1, 0


def run_backtest(  # noqa: PLR0915 — the loop mirrors the eight numbered steps of §9.2 literally;
    # splitting it across helpers would hide the ordering this spec is most particular about.
    bars: pl.DataFrame,
    targets: pl.DataFrame,
    config: BacktestConfig,
    *,
    funding: pl.DataFrame | None = None,
    risk: RiskEngine | None = None,
    interval_ms: int | None = None,
) -> BacktestResult:
    """Run the §9.2 event loop over `bars`, following `targets` with a one-bar execution delay.

    `bars` needs open_time/open/close/quote_volume. `targets` needs timestamp/target_position, where
    a target stamped at bar `t` was computed from bar `t` close and therefore fills at `t+1` open.
    """
    bars = bars.sort("open_time")
    if bars.height < 2:
        raise ValueError("a backtest needs at least two bars — fills happen at the next bar's open")

    regime = config.regime
    risk = risk or RiskEngine()

    # The one-bar delay, made structural rather than remembered: a target stamped at t is joined
    # onto bar t+1, so the loop below physically cannot see a same-bar signal.
    aligned = (
        bars.join(
            targets.select(pl.col("timestamp"), pl.col("target_position").cast(pl.Float64)).sort("timestamp"),
            left_on="open_time",
            right_on="timestamp",
            how="left",
        )
        .with_columns(pl.col("target_position").shift(1).fill_null(0.0).alias("effective_target"))
        .drop("target_position")
    )

    funding_map: dict[int, float] = {}
    if funding is not None and funding.height:
        funding_map = dict(
            zip(
                funding["funding_time"].to_list(),
                funding["funding_rate"].to_list(),
                strict=True,
            )
        )
    funding_times = np.array(sorted(funding_map), dtype=np.int64)

    state = PortfolioState(cash=config.initial_equity)
    records: list[dict[str, Any]] = []
    capacity_breaches = 0
    partial_fills = 0
    bankrupt = False
    prev_equity = config.initial_equity

    open_times = aligned["open_time"].to_numpy()
    opens = aligned["open"].to_numpy()
    closes = aligned["close"].to_numpy()
    volumes = aligned["quote_volume"].to_numpy()
    effective = aligned["effective_target"].to_numpy()
    step = int(interval_ms or (open_times[1] - open_times[0]))

    for i in range(aligned.height):
        t = int(open_times[i])
        fill_price = float(opens[i])
        mark_price = float(closes[i])
        funding_flow = 0.0

        # 1. apply funding if t is a settlement timestamp
        if funding_times.size:
            due = funding_times[(funding_times > t - step) & (funding_times <= t)]
            for settlement in due:
                funding_flow += apply_funding(state, fill_price, funding_map[int(settlement)], regime)

        # 2. mark existing positions to market
        equity_now = state.equity(mark_price)
        if equity_now <= 0:
            # The account is gone. Continuing to trade from negative equity produces a P&L path
            # that means nothing, so the run ends here and the result is flagged.
            bankrupt = True

        # 3. check the risk engine — may force flatten. Realised exposure is passed so the
        # gross-leverage limit is enforced against the actual book, not just the target.
        risk.observe(t, equity_now)
        forced_flat = risk.check(t, equity_now, gross_notional=state.position.notional(mark_price))

        # 4. read target_position (computed at t-1 close), 5. diff, 6. no-trade band
        #
        # Both sides of the diff are in *target* units, where 1.0 means `max_leverage` times equity.
        # Comparing a target in [-1, 1] against a raw leverage ratio would make a fully-invested
        # position read back as 2.0 and churn the book every bar.
        warm = i < config.warmup_bars
        desired = 0.0 if (forced_flat or warm or bankrupt) else risk.clamp(float(effective[i]))
        denominator = equity_now * config.max_leverage
        current_target = state.position.notional(fill_price) / denominator if denominator > 0 else 0.0
        band = 0.0 if forced_flat else config.no_trade_band
        delta_target = no_trade_band_filter(desired, current_target, band)

        # 7. fill at t open + slippage + fees
        traded_notional = 0.0
        if delta_target != 0.0 and equity_now > 0 and fill_price > 0:
            delta_units = delta_target * denominator / fill_price
            bar_volume = float(volumes[i])
            capped, breached, partial = _apply_capacity_limit(
                delta_units, state.position.units, fill_price, bar_volume, config
            )
            capacity_breaches += breached
            partial_fills += partial
            delta_units = capped

            if delta_units != 0.0:
                state, cost = apply_fill(
                    state,
                    delta_units,
                    fill_price,
                    bar_volume,
                    regime,
                    half_spread_bps=config.half_spread_bps,
                    maker=config.maker,
                    impact_k=config.impact_k,
                    max_participation=config.max_participation,
                )
                traded_notional = cost.notional

        # 8. record state
        equity_end = state.equity(mark_price)
        records.append(
            {
                "open_time": t,
                "price": mark_price,
                "target_position": float(effective[i]),
                "position_units": state.position.units,
                "exposure": state.position.notional(mark_price) / equity_end if equity_end > 0 else 0.0,
                "cash": state.cash,
                "equity": equity_end,
                "traded_notional": traded_notional,
                "realised_costs": state.realised_costs,
                "funding_flow": funding_flow,
                "return": (equity_end - prev_equity) / prev_equity if prev_equity > 0 else 0.0,
                "equity_gross": equity_end + state.realised_costs + state.funding_paid,
                "killed": risk.kill_switch.tripped,
                "bankrupt": bankrupt,
            }
        )
        prev_equity = equity_end
        if bankrupt:
            break

    curve = pl.DataFrame(records)
    net_pnl = float(curve["equity"][-1]) - config.initial_equity
    fees_and_slippage = state.realised_costs
    gross_pnl = net_pnl + fees_and_slippage + state.funding_paid
    span_years = max((int(open_times[-1]) - int(open_times[0])) / MS_PER_YEAR, 1e-9)
    turnover = float(curve["traded_notional"].sum()) / config.initial_equity / span_years

    return BacktestResult(
        equity_curve=curve,
        config=config,
        gross_pnl=gross_pnl,
        fees_and_slippage=fees_and_slippage,
        funding_paid=state.funding_paid,
        net_pnl=net_pnl,
        turnover_per_year=turnover,
        capacity_breaches=capacity_breaches,
        partial_fills=partial_fills,
        kill_reason=risk.kill_switch.reason.value if risk.kill_switch.reason else None,
        bars=curve.height,
        interval_ms=step,
    )
