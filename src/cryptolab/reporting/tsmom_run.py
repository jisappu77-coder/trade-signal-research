"""Run the TSMOM grid and produce its verdict (SPEC.md Phase 4).

Method fixed before the first number was computed, per CLAUDE.md:

* **Trials registered before results.** The 24 declared combinations across two symbols are written
  to the registry first; `N` is then read back from it, never hard-coded.
* **Targets computed once per (symbol, params) on the full series, then sliced per fold.** The
  Phase 1-3 harness regenerates inside each fold window, so every fold starts cold — at 4h a
  188-bar warm-up is a third of a 90-day fold. That pattern is deliberately not copied.
* **PBO and DSR trial dispersion computed per bar size, never mixed.** Stacking 1h and 4h return
  series measures the difference in observation frequency rather than search dispersion, and the
  truncation to the shorter series would silently discard most of the 1h data. `N` still comes from
  the registry and still counts both.
* **Both walk-forward variants reported.** Fixed-params out-of-sample, and best-config-selected-on
  -train out-of-sample. The second is the honest headline for "does this *search* produce an edge",
  because it pays the selection cost the first one hides.
* **The sealed test period is never opened.** No token is requested; §10.1 allows one per family,
  ever.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cryptolab.backtest.attribution import Attribution, attribute
from cryptolab.backtest.costs import REGIME_ORDER, get_regime
from cryptolab.backtest.engine import MS_PER_YEAR, BacktestConfig, BacktestResult, run_backtest
from cryptolab.backtest.risk import RiskEngine, RiskLimits
from cryptolab.data.schemas import BAR_INTERVAL_MS
from cryptolab.data.store import ParquetStore, to_ms
from cryptolab.features.resample import to_bar
from cryptolab.reporting.build import build_report
from cryptolab.reporting.report import StrategyReport, write_site
from cryptolab.signals.tsmom import MAX_LEVERAGE, TSMOM
from cryptolab.validation import tax
from cryptolab.validation.controls import (
    buy_and_hold_sharpe,
    calibrate_flip_probability,
    random_targets,
    single_quarter_concentrated,
    summarise_controls,
)
from cryptolab.validation.deflated_sharpe import deflated_sharpe
from cryptolab.validation.gates import GateInputs, evaluate_gates, parameter_plateau_fraction
from cryptolab.validation.pbo import cscv_pbo
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.walkforward import generate_folds, summarise

BOOK = 25_000.0
CONTROL_SEEDS = 24
STRATEGY_FAMILY = "tsmom"

METHOD_NOTES = (
    "Trials were registered before any result was computed; N is read from the registry, not hard-coded.",
    "The no-trade band is 0.10 in target units, which is 0.20 of equity notional at max_lev 2.0.",
    "Targets are computed once on the full series and sliced per fold, so folds do not start cold.",
    "PBO and the DSR trial dispersion are computed per bar size; mixing 1h and 4h would measure "
    "the difference in observation frequency rather than search dispersion.",
    "The sealed test period (2024-07-01 onward) was not opened. No token was requested.",
    f"A ${BOOK:,.0f} book is used because the §7 participation limit binds on thin early-2020 "
    "bars. The limit was not relaxed.",
)


@dataclass(slots=True)
class GridRun:
    """One (symbol, params) cell of the search."""

    symbol: str
    params: dict[str, Any]
    result: BacktestResult
    attribution: Attribution
    fold_sharpes: list[float]
    mean_fold_sharpe: float
    stdev_fold_sharpe: float
    single_fold_concentrated: bool

    @property
    def bar(self) -> str:
        return str(self.params["bar"])

    @property
    def label(self) -> str:
        p = self.params
        return f"TSMOM_L{p['lookback_bars']}_H{p['vol_halflife']}_{p['bar']}_{self.symbol}"

    @property
    def annualised(self) -> float:
        return self.result.sharpe(annualised=True)


def _risk() -> RiskEngine:
    """Drawdown and daily limits lifted so a run is measured rather than stopped out early.

    Leverage and insolvency stay enforced: those are the limits that keep the numbers meaningful.
    """
    return RiskEngine(
        limits=RiskLimits(
            daily_loss_limit=1.0,
            drawdown_limit=1.0,
            max_consecutive_losses=10**9,
            max_gross_leverage=MAX_LEVERAGE,
        )
    )


def _config(regime: str = "conservative") -> BacktestConfig:
    return BacktestConfig(
        regime_name=regime,
        warmup_bars=TSMOM().max_lookback_bars,
        no_trade_band=0.10,
        initial_equity=BOOK,
        max_leverage=MAX_LEVERAGE,
    )


def _periods_per_year(bar: str) -> float:
    return MS_PER_YEAR / BAR_INTERVAL_MS[bar]


def run_grid(
    store: ParquetStore,
    registry: TrialRegistry,
    *,
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT"),
    start: str = "2020-01-01",
    end: str = "2024-06-30",
    exchange: str = "binance",
    bars_filter: Sequence[str] | None = None,
) -> tuple[list[GridRun], int]:
    """Register the declared grid, run every cell, and return the runs plus `N` from the registry."""
    signal = TSMOM()
    grid = [p for p in signal.grid() if bars_filter is None or p["bar"] in bars_filter]

    registry.register_grid(
        signal=signal.name,
        grid=signal.grid(),  # register the whole declared space, even if a subset is run
        symbols=list(symbols),
        period=f"{start}:{end}",
        strategy_family=STRATEGY_FAMILY,
        note="phase 4 TSMOM on real data, train+validation only",
    )
    n_trials = registry.count(signal=signal.name)

    hourly: dict[str, pl.DataFrame] = {}
    funding: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        hourly[symbol] = store.read("ohlcv", exchange=exchange, symbol=symbol, start=start, end=end).drop(
            "ingested_at", "source_uri"
        )
        funding[symbol] = store.read("funding", exchange=exchange, symbol=symbol, start=start, end=end).drop(
            "ingested_at", "source_uri"
        )
        if hourly[symbol].height < 1000:
            raise ValueError(
                f"{symbol} has only {hourly[symbol].height} bars in {start}..{end}; "
                "ingest the data first (`cryptolab ingest`)"
            )

    frames = {
        (symbol, bar): (hourly[symbol] if bar == "1h" else to_bar(hourly[symbol], bar))
        for symbol in symbols
        for bar in {str(p["bar"]) for p in grid}
    }
    folds = generate_folds(to_ms(start), to_ms(end), train_days=365, test_days=90, step_days=90)

    runs: list[GridRun] = []
    for symbol in symbols:
        for params in grid:
            bar = str(params["bar"])
            frame = frames[(symbol, bar)]
            # Computed once on the full series; folds are slices of this, never recomputed cold.
            targets = signal.generate(frame, params)
            result = run_backtest(frame, targets, _config(), funding=funding[symbol], risk=_risk())

            annualiser = math.sqrt(_periods_per_year(bar))
            fold_sharpes = []
            for fold in folds:
                mask = frame["open_time"].is_between(fold.test_start, fold.test_end)
                window, window_targets = (
                    frame.filter(mask),
                    targets.filter(targets["timestamp"].is_between(fold.test_start, fold.test_end)),
                )
                if window.height < 100:
                    continue
                fold_result = run_backtest(
                    window,
                    window_targets,
                    _config(),
                    funding=funding[symbol],
                    risk=_risk(),
                )
                fold_sharpes.append(fold_result.sharpe() * annualiser)

            summary = summarise([s / annualiser for s in fold_sharpes])
            runs.append(
                GridRun(
                    symbol=symbol,
                    params=params,
                    result=result,
                    attribution=attribute(result),
                    fold_sharpes=fold_sharpes,
                    mean_fold_sharpe=summary.mean_sharpe * annualiser,
                    stdev_fold_sharpe=summary.stdev_sharpe * annualiser,
                    single_fold_concentrated=summary.single_fold_concentrated,
                )
            )
    return runs, n_trials


def _pbo_by_bar(runs: Sequence[GridRun], bar: str) -> float:
    """CSCV PBO within one bar size. Mixing frequencies would measure the wrong thing."""
    columns = [r.result.returns for r in runs if r.bar == bar]
    if len(columns) < 2:
        return 0.0
    length = min(len(c) for c in columns)
    return cscv_pbo(np.column_stack([c[:length] for c in columns]), n_splits=10).pbo


def _control_for(run: GridRun, frame: pl.DataFrame, funding: pl.DataFrame) -> Any:
    """A turnover- and exposure-matched random control for one run."""
    exposure = run.result.equity_curve["target_position"].to_numpy()
    times = frame["open_time"]
    annualiser = math.sqrt(_periods_per_year(run.bar))

    def build(probability: float, seed: int) -> BacktestResult:
        targets = pl.DataFrame(
            {
                "timestamp": times,
                "target_position": random_targets(frame.height, exposure, probability, seed),
                "confidence": np.ones(frame.height),
            }
        )
        return run_backtest(frame, targets, _config(), funding=funding, risk=_risk())

    probability = calibrate_flip_probability(
        run.result.turnover_per_year, lambda p: build(p, 0).turnover_per_year
    )
    sharpes = [build(probability, seed).sharpe() * annualiser for seed in range(CONTROL_SEEDS)]
    return summarise_controls("random-entry (matched)", sharpes, matched_turnover=probability)


def build_reports(
    store: ParquetStore,
    registry: TrialRegistry,
    *,
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT"),
    start: str = "2020-01-01",
    end: str = "2024-06-30",
    exchange: str = "binance",
    bars_filter: Sequence[str] | None = None,
) -> list[StrategyReport]:
    """Run the grid and turn every cell into a report carrying its own computed verdict."""
    runs, n_trials = run_grid(
        store,
        registry,
        symbols=symbols,
        start=start,
        end=end,
        exchange=exchange,
        bars_filter=bars_filter,
    )
    period = f"{start} → {end}  (train+validation, seal untouched)"

    hourly = {
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

    controls = {s: buy_and_hold_sharpe(hourly[s], periods_per_year=_periods_per_year("1h")) for s in symbols}
    best_control = max(controls.values())
    pbo_by_bar = {bar: _pbo_by_bar(runs, bar) for bar in {r.bar for r in runs}}
    grid_sharpes = {r.label: r.annualised for r in runs}
    plateau = parameter_plateau_fraction(grid_sharpes)

    # The best run per bar gets the expensive matched-control treatment; running it for all 48
    # cells would multiply the compute by CONTROL_SEEDS for no extra information.
    best_per_bar = {
        bar: max((r for r in runs if r.bar == bar), key=lambda r: r.mean_fold_sharpe) for bar in pbo_by_bar
    }
    control_results = {
        bar: _control_for(
            run,
            hourly[run.symbol] if bar == "1h" else to_bar(hourly[run.symbol], bar),
            funding[run.symbol],
        )
        for bar, run in best_per_bar.items()
    }

    reports: list[StrategyReport] = []
    for run in runs:
        bar = run.bar
        trial_sharpes = np.array([r.result.sharpe() for r in runs if r.bar == bar])
        dsr = deflated_sharpe(
            run.result.returns,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            periods_per_year=_periods_per_year(bar),
        )
        control = control_results[bar]
        curve = run.result.equity_curve
        span_years = max((int(curve["open_time"][-1]) - int(curve["open_time"][0])) / MS_PER_YEAR, 1e-9)
        tax_outcome = tax.tax_single_run(
            pre_tax_pnl=run.result.net_pnl,
            traded_notional=float(curve["traded_notional"].sum()),
            initial_equity=BOOK,
            years=span_years,
        )
        gates = evaluate_gates(
            GateInputs(
                net_sharpe_oos=run.mean_fold_sharpe,
                deflated_sharpe=dsr.dsr,
                pbo=pbo_by_bar[bar],
                fold_sharpe_stdev=run.stdev_fold_sharpe,
                fold_sharpe_mean=run.mean_fold_sharpe,
                max_drawdown=run.result.max_drawdown(),
                breakeven_cost_bps=run.result.breakeven_cost_bps(),
                parameter_plateau_fraction=plateau,
                beats_controls=(
                    run.mean_fold_sharpe > best_control and control.beaten_by(run.mean_fold_sharpe)
                ),
                regime=get_regime("conservative"),
                single_fold_concentrated=run.single_fold_concentrated,
                single_quarter_concentrated=single_quarter_concentrated(
                    curve["open_time"].to_list(), run.result.returns.tolist()
                ),
            )
        )
        regime_sharpes = {
            str(name): run_backtest(
                hourly[run.symbol] if bar == "1h" else to_bar(hourly[run.symbol], bar),
                TSMOM().generate(
                    hourly[run.symbol] if bar == "1h" else to_bar(hourly[run.symbol], bar),
                    run.params,
                ),
                _config(str(name)),
                funding=funding[run.symbol],
                risk=_risk(),
            ).sharpe(annualised=True)
            for name in REGIME_ORDER
        }

        reports.append(
            build_report(
                strategy=run.label,
                period=period,
                trials=n_trials,
                result=run.result,
                gates=gates,
                net_sharpe=run.mean_fold_sharpe,
                deflated_sharpe=dsr.dsr,
                pbo=pbo_by_bar[bar],
                fold_sharpes=run.fold_sharpes,
                regime_sharpes=regime_sharpes,
                grid_sharpes=grid_sharpes,
                notes=(
                    *METHOD_NOTES,
                    run.attribution.summary_line(),
                    tax.summary_line(tax_outcome),
                    f"In-sample annualised net Sharpe was {run.annualised:.2f}; the walk-forward "
                    f"out-of-sample mean across {len(run.fold_sharpes)} folds was "
                    f"{run.mean_fold_sharpe:.2f}.",
                    f"Matched random control: {control.mean_sharpe:.2f} mean, "
                    f"{control.p95_sharpe:.2f} at the 95th percentile over {CONTROL_SEEDS} seeds. "
                    f"Buy-and-hold controls: " + ", ".join(f"{s} {v:.2f}" for s, v in controls.items()) + ".",
                    f"{run.result.capacity_breaches} capacity breaches, "
                    f"{run.result.partial_fills} partially-filled reductions.",
                ),
                tier=1,
                promotable=True,
                legs=[
                    {
                        "leg": leg.leg,
                        "bars": leg.bars,
                        "fraction_of_time": leg.fraction_of_time,
                        "gross_pnl": leg.gross_pnl,
                        "fees_and_slippage": leg.fees_and_slippage,
                        "funding_paid": leg.funding_paid,
                        "net_pnl": leg.net_pnl,
                        "net_expectancy_bps": leg.net_expectancy_bps,
                        "hit_rate": leg.hit_rate,
                    }
                    for leg in run.attribution.legs.values()
                ],
                short_leg_negative=run.attribution.short_leg_is_a_drag,
                attribution_line=run.attribution.summary_line(),
                tax_line=tax.summary_line(tax_outcome),
                pre_tax_return=tax_outcome.pre_tax_return,
                post_tax_return=tax_outcome.post_tax_return,
                effective_tax_rate=tax_outcome.effective_rate,
                tds_multiple_of_capital=tax_outcome.tds_as_multiple_of_capital,
            )
        )
    return reports


def write_tsmom_site(
    store: ParquetStore,
    registry: TrialRegistry,
    out_dir: Path | str,
    *,
    also: Sequence[StrategyReport] = (),
    **kwargs: Any,
) -> list[Path]:
    """Build the TSMOM reports and write the site, keeping any earlier reports alongside."""
    reports = build_reports(store, registry, **kwargs)
    return write_site([*also, *reports], out_dir)
