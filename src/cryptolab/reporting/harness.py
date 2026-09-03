"""Build the Phase 1–3 validation site from real data.

This exists because there is nothing else to report on yet: no Tier-1 signal is built, so the only
honest content for the index is the harness validation itself. Each run here uses `MomentumProbe`,
which is **Tier 3 and non-promotable** — it exercises the instrument and is labelled as such in
every report it produces.

At Phase 4 this module is replaced by real strategy runs. The reporting layer it drives does not
change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cryptolab.backtest.costs import REGIME_ORDER, get_regime
from cryptolab.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from cryptolab.backtest.risk import RiskEngine, RiskLimits
from cryptolab.data.store import ParquetStore, to_ms
from cryptolab.reporting.build import build_report
from cryptolab.reporting.report import StrategyReport, write_site
from cryptolab.validation.deflated_sharpe import deflated_sharpe
from cryptolab.validation.gates import GateInputs, evaluate_gates, parameter_plateau_fraction
from cryptolab.validation.pbo import cscv_pbo
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.synthetic import MomentumProbe
from cryptolab.validation.walkforward import generate_folds, summarise

BARS_PER_YEAR = 8766.0
BOOK = 25_000.0

NOTES = (
    "MomentumProbe is Tier 3 and non-promotable. It exists to exercise the harness; its numbers "
    "are not a research result.",
    "Full-sample figures span train and validation together. They are shown only as the in-sample "
    "counterpart to the walk-forward number.",
    "The sealed test period (2024-07-01 onward) was never opened for this run.",
    "A $25,000 book is used because the §7 participation limit binds on thin early-2020 bars. The "
    "limit was not relaxed to fit a larger book.",
)


def _risk() -> RiskEngine:
    """Drawdown and daily limits lifted so a run is measured rather than stopped out early.

    Leverage and insolvency stay enforced — those are the limits that keep the numbers meaningful.
    """
    return RiskEngine(
        limits=RiskLimits(
            daily_loss_limit=1.0,
            drawdown_limit=1.0,
            max_consecutive_losses=10**9,
            max_gross_leverage=2.0,
        )
    )


def build_harness_site(
    store: ParquetStore,
    registry: TrialRegistry,
    out_dir: Path | str,
    *,
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT"),
    start: str = "2020-01-01",
    end: str = "2024-06-30",
    exchange: str = "binance",
) -> list[Path]:
    """Run the probe grid over real data and write the report site. Returns the paths written."""
    probe = MomentumProbe()
    grid = probe.grid()
    period = f"{start} → {end}  (train+validation, seal untouched)"

    bars = {
        s: store.read("ohlcv", exchange=exchange, symbol=s, start=start, end=end).drop(
            "ingested_at", "source_uri"
        )
        for s in symbols
    }
    funding = {
        s: store.read("funding", exchange=exchange, symbol=s, start=start, end=end).drop(
            "ingested_at", "source_uri"
        )
        for s in symbols
    }
    for symbol, frame in bars.items():
        if frame.height < 500:
            raise ValueError(
                f"{symbol} has only {frame.height} bars in {start}..{end}; ingest the data first "
                "(`cryptolab ingest`)"
            )

    registry.register_grid(
        signal=probe.name,
        grid=grid,
        symbols=list(symbols),
        period=f"{start}:{end}",
        note="phase 1-3 harness validation on real data; tier 3, not promotable",
    )
    n_trials = registry.count(signal=probe.name)

    def run(
        symbol: str,
        params: dict[str, Any],
        regime: str = "conservative",
        data: pl.DataFrame | None = None,
    ) -> BacktestResult:
        frame = bars[symbol] if data is None else data
        targets = probe.generate(frame, params)
        return run_backtest(
            frame,
            targets,
            BacktestConfig(regime_name=regime, warmup_bars=200, no_trade_band=0.10, initial_equity=BOOK),
            funding=funding[symbol],
            risk=_risk(),
        )

    results = {(s, int(p["lookback"])): run(s, dict(p)) for s in symbols for p in grid}
    trial_sharpes = np.array([r.sharpe() for r in results.values()])
    grid_sharpes = {f"{s[:3]} L{lb}": r.sharpe(annualised=True) for (s, lb), r in results.items()}
    plateau = parameter_plateau_fraction(grid_sharpes)

    length = min(len(r.returns) for r in results.values())
    pbo = cscv_pbo(np.column_stack([r.returns[:length] for r in results.values()]), n_splits=10)

    def buy_hold(symbol: str) -> float:
        close = bars[symbol]["close"].to_numpy()
        rets = np.diff(close) / close[:-1]
        return float(np.mean(rets) / np.std(rets, ddof=1)) * math.sqrt(BARS_PER_YEAR)

    controls = max(buy_hold(s) for s in symbols)
    folds = generate_folds(to_ms(start), to_ms(end), train_days=365, test_days=90, step_days=90)
    reports: list[StrategyReport] = []

    for (symbol, lookback), result in results.items():
        params: dict[str, Any] = {"lookback": lookback}
        fold_sharpes = []
        for fold in folds:
            window = bars[symbol].filter(pl.col("open_time").is_between(fold.test_start, fold.test_end))
            fold_sharpes.append(
                run(symbol, params, data=window).sharpe() * math.sqrt(BARS_PER_YEAR)
                if window.height >= 300
                else 0.0
            )
        summary = summarise([s / math.sqrt(BARS_PER_YEAR) for s in fold_sharpes])
        mean_ann = summary.mean_sharpe * math.sqrt(BARS_PER_YEAR)
        stdev_ann = summary.stdev_sharpe * math.sqrt(BARS_PER_YEAR)

        dsr = deflated_sharpe(
            result.returns,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            periods_per_year=BARS_PER_YEAR,
        )
        regime_sharpes: dict[str, float] = {
            str(name): run(symbol, params, regime=name).sharpe(annualised=True) for name in REGIME_ORDER
        }
        gates = evaluate_gates(
            GateInputs(
                net_sharpe_oos=mean_ann,
                deflated_sharpe=dsr.dsr,
                pbo=pbo.pbo,
                fold_sharpe_stdev=stdev_ann,
                fold_sharpe_mean=mean_ann,
                max_drawdown=result.max_drawdown(),
                breakeven_cost_bps=result.breakeven_cost_bps(),
                parameter_plateau_fraction=plateau,
                beats_controls=mean_ann > controls,
                regime=get_regime("conservative"),
                single_fold_concentrated=summary.single_fold_concentrated,
            )
        )
        reports.append(
            build_report(
                strategy=f"PROBE_L{lookback}_1h_{symbol}",
                period=period,
                trials=n_trials,
                result=result,
                gates=gates,
                net_sharpe=mean_ann,
                deflated_sharpe=dsr.dsr,
                pbo=pbo.pbo,
                fold_sharpes=fold_sharpes,
                regime_sharpes=regime_sharpes,
                grid_sharpes=grid_sharpes,
                notes=(
                    *NOTES,
                    f"Full-sample annualised net Sharpe was {result.sharpe(annualised=True):.2f}; "
                    f"the walk-forward out-of-sample mean was {mean_ann:.2f}. That gap is the "
                    "point of this instrument.",
                    f"{result.capacity_breaches} capacity breaches and {result.partial_fills} "
                    "partially-filled reductions on this path.",
                ),
                tier=3,
                promotable=False,
            )
        )

    return write_site(reports, out_dir)
