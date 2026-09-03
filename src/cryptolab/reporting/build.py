"""Assemble StrategyReport objects from a completed backtest and its validation results."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np

from cryptolab.backtest.costs import get_regime
from cryptolab.backtest.engine import BacktestResult
from cryptolab.data.store import from_ms
from cryptolab.reporting.report import SeriesPoint, StrategyReport
from cryptolab.validation.gates import GateReport

MAX_POINTS = 260


def downsample(result: BacktestResult, *, max_points: int = MAX_POINTS) -> list[SeriesPoint]:
    """Thin the equity curve for rendering.

    Takes evenly spaced samples and always keeps the final bar, so the endpoint a reader checks
    against the reported P&L is the real one rather than whichever sample landed nearby.
    """
    curve = result.equity_curve
    if curve.height == 0:
        return []
    idx = sorted({*np.linspace(0, curve.height - 1, min(max_points, curve.height)).astype(int).tolist()})
    times = curve["open_time"].to_numpy()
    net = curve["equity"].to_numpy()
    gross = curve["equity_gross"].to_numpy()
    return [
        SeriesPoint(
            label=from_ms(int(times[i])).strftime("%Y-%m-%d"),
            net=float(net[i]),
            gross=float(gross[i]),
        )
        for i in idx
    ]


def build_report(
    *,
    strategy: str,
    period: str,
    trials: int,
    result: BacktestResult,
    gates: GateReport,
    net_sharpe: float,
    deflated_sharpe: float,
    pbo: float,
    fold_sharpes: Sequence[float] = (),
    regime_sharpes: dict[str, float] | None = None,
    grid_sharpes: dict[str, float] | None = None,
    notes: Sequence[str] = (),
    tier: int = 1,
    promotable: bool = True,
) -> StrategyReport:
    """Assemble one report. Every displayed number comes from the objects passed in."""
    return StrategyReport(
        strategy=strategy,
        period=period,
        trials=trials,
        regime=get_regime(result.config.regime_name),
        net_sharpe=net_sharpe,
        deflated_sharpe=deflated_sharpe,
        pbo=pbo,
        breakeven_bps=result.breakeven_cost_bps(),
        turnover=result.turnover_per_year,
        cost_drag_bps=result.cost_drag_bps_per_year(),
        max_drawdown=result.max_drawdown(),
        gates=gates,
        flat_fraction=result.flat_fraction,
        kill_reason=result.kill_reason,
        killed_from=(
            from_ms(result.first_kill_time).strftime("%Y-%m-%d")
            if result.first_kill_time is not None
            else None
        ),
        equity=downsample(result),
        fold_sharpes=list(fold_sharpes),
        regime_sharpes=regime_sharpes or {},
        grid_sharpes=grid_sharpes or {},
        notes=list(notes),
        tier=tier,
        promotable=promotable,
        generated_at=dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
