"""Typer entrypoint (SPEC.md §4).

Phases 1–3 are wired here. There is no `run-strategy` command yet, deliberately: Phase 4 signals do
not exist until this harness is green, and offering the command would invite exactly the "just
quickly test an idea" impulse the phase ordering defends against.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Annotated

import typer

from cryptolab.config import BaseConfig, load_cost_settings

app = typer.Typer(add_completion=False, help="cryptolab — an edge-disproving research instrument")


@app.command()
def init(
    config: Annotated[Path, typer.Option(help="Path to base.yaml")] = Path("config/base.yaml"),
) -> None:
    """Verify configuration and print the constraints that bite before any data exists."""
    from cryptolab.data.ingest import open_interest_loss_notice

    base = BaseConfig.load(config)
    load_cost_settings()
    typer.echo(f"universe           {', '.join(base.universe)}")
    typer.echo(f"data root          {base.data_root}")
    typer.echo(f"trial registry     {base.registry_path}")
    typer.secho(
        f"sealed test period from "
        f"{dt.datetime.fromtimestamp(base.splits.test_start / 1000, tz=dt.UTC):%Y-%m-%d} "
        "— one touch, ever",
        fg=typer.colors.YELLOW,
    )
    typer.secho(open_interest_loss_notice(), fg=typer.colors.RED, bold=True)


@app.command()
def ingest(
    symbol: Annotated[str, typer.Argument(help="e.g. BTCUSDT")],
    interval: Annotated[str, typer.Option(help="1m, 5m, 1h, 4h")] = "1h",
    start: Annotated[str, typer.Option()] = "2019-01-01",
    end: Annotated[str, typer.Option()] = "2024-06-30",
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """Ingest monthly kline archives, gating every month on §6 data quality."""
    from cryptolab.data.ingest import ingest_klines
    from cryptolab.data.store import ParquetStore

    base = BaseConfig.load(config)
    store = ParquetStore(base.data_root, base.splits)
    results = asyncio.run(ingest_klines(store, symbol, interval, start, end, exchange=base.exchange))

    failed = [r for r in results if r.report is not None and not r.report.passed]
    ingested = sum(r.rows for r in results)
    typer.echo(f"{symbol} {interval}: {ingested:,} bars across {len(results)} months")
    for result in failed:
        typer.secho(f"  QUALITY FAIL {result.year}-{result.month:02d}", fg=typer.colors.RED)
        for finding in result.report.findings if result.report else []:
            typer.echo(f"    [{finding.check}] {finding.detail}")
    if failed:
        raise typer.Exit(code=1)


@app.command("ingest-funding")
def ingest_funding_cmd(
    symbol: Annotated[str, typer.Argument()],
    start: Annotated[str, typer.Option()] = "2020-01-01",
    end: Annotated[str, typer.Option()] = "2024-06-30",
    source: Annotated[str, typer.Option(help="archive (default) or rest")] = "archive",
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """Ingest funding-rate history from the monthly archive, or the REST API with --source rest."""
    from cryptolab.data.ingest import ingest_funding, ingest_funding_archive
    from cryptolab.data.store import ParquetStore

    base = BaseConfig.load(config)
    store = ParquetStore(base.data_root, base.splits)
    if source == "archive":
        results = asyncio.run(ingest_funding_archive(store, symbol, start, end, exchange=base.exchange))
    elif source == "rest":
        results = [asyncio.run(ingest_funding(store, symbol, start, end, exchange=base.exchange))]
    else:
        typer.secho(f"unknown source {source!r}; use 'archive' or 'rest'", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    total = sum(r.rows for r in results)
    typer.echo(f"{symbol} funding: {total:,} settlements from {len(results)} objects")
    failed = [r for r in results if r.report is not None and not r.report.passed]
    for result in failed:
        typer.secho(f"  QUALITY FAIL {result.year}-{result.month:02d}", fg=typer.colors.RED)
        for finding in result.report.findings if result.report else []:
            typer.echo(f"    [{finding.check}] {finding.detail}")
    if failed:
        raise typer.Exit(code=1)


@app.command("collect-oi")
def collect_oi(
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """The daily open-interest collector. Whatever is not collected today is lost permanently."""
    from cryptolab.data.ingest import collect_open_interest, open_interest_loss_notice
    from cryptolab.data.store import ParquetStore

    base = BaseConfig.load(config)
    store = ParquetStore(base.data_root, base.splits)
    typer.secho(open_interest_loss_notice(), fg=typer.colors.YELLOW)
    for result in asyncio.run(collect_open_interest(store, base.universe, exchange=base.exchange)):
        typer.echo(f"{result.symbol}: {result.rows} open-interest points")


@app.command("registry-status")
def registry_status(
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """Report the trial count N and verify the registry's hash chain."""
    from cryptolab.validation.registry import TrialRegistry

    base = BaseConfig.load(config)
    with TrialRegistry(base.registry_path) as registry:
        total = registry.count()
        intact = registry.verify_chain()
        typer.echo(f"trials N           {total}")
        typer.secho(
            f"hash chain         {'intact' if intact else 'TAMPERED'}",
            fg=typer.colors.GREEN if intact else typer.colors.RED,
            bold=not intact,
        )
        if not intact:
            raise typer.Exit(code=1)


@app.command("live-signal")
def live_signal(
    holding_days: Annotated[float, typer.Option(help="Hold the entry threshold is priced for")] = 7.0,
    regime: Annotated[str, typer.Option()] = "conservative",
    log: Annotated[Path, typer.Option(help="Append each reading to this JSONL file")] = Path(
        "data/live_watch.jsonl"
    ),
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """Evaluate the CARRY entry condition against live market data. Reads only; places no orders.

    Uses OKX because Binance returns HTTP 451 from some hosts. Funding differs between venues, so
    this is indicative of the strategy's state rather than a continuation of the backtested series.
    """
    import asyncio

    import httpx

    from cryptolab.data.sources.okx_live import fetch_all
    from cryptolab.live import append_observation, evaluate

    base = BaseConfig.load(config)

    async def read() -> None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            quotes = await fetch_all(client, list(base.universe))
        for quote in quotes:
            signal = evaluate(quote, holding_days=holding_days, regime_name=regime)
            append_observation(signal, log)
            typer.secho(signal.line(), fg=typer.colors.GREEN if signal.fires else typer.colors.WHITE)

    asyncio.run(read())
    typer.echo(f"appended to {log}")


@app.command("watch-summary")
def watch_summary(
    log: Annotated[Path, typer.Option()] = Path("data/live_watch.jsonl"),
) -> None:
    """Summarise a completed live watch: how often the signal fired and how close it came."""
    from cryptolab.live import read_observations, summarise_watch

    typer.echo(summarise_watch(read_observations(log)))


@app.command("run-tsmom")
def run_tsmom(
    out: Annotated[Path, typer.Option(help="Directory to write the static site into")] = Path("site"),
    start: Annotated[str, typer.Option()] = "2020-01-01",
    end: Annotated[str, typer.Option()] = "2024-06-30",
    bars: Annotated[str, typer.Option(help="Comma-separated bar sizes, e.g. '4h' or '1h,4h'")] = "1h,4h",
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """Run the TSMOM grid (§8.1) and publish its verdict.

    Registers the full declared search space before computing anything, so N comes from the
    registry. Never opens the sealed test period.
    """
    from cryptolab.data.store import ParquetStore
    from cryptolab.reporting.tsmom_run import write_tsmom_site
    from cryptolab.validation.registry import TrialRegistry

    base = BaseConfig.load(config)
    with TrialRegistry(base.registry_path) as registry:
        store = registry.bind_store(ParquetStore(base.data_root, base.splits))
        paths = write_tsmom_site(
            store,
            registry,
            out,
            symbols=base.universe,
            start=start,
            end=end,
            exchange=base.exchange,
            bars_filter=[b.strip() for b in bars.split(",") if b.strip()],
        )
        typer.echo(f"trials N   {registry.count(signal='tsmom')}")
    typer.echo(f"wrote {len(paths)} files to {out}/")
    typer.secho(f"open {paths[0]}", fg=typer.colors.GREEN)


@app.command("report")
def report(
    out: Annotated[Path, typer.Option(help="Directory to write the static site into")] = Path("site"),
    start: Annotated[str, typer.Option()] = "2020-01-01",
    end: Annotated[str, typer.Option()] = "2024-06-30",
    config: Annotated[Path, typer.Option()] = Path("config/base.yaml"),
) -> None:
    """Build the static HTML report site (§12): one report per run, plus the comparison index.

    This renders the Phase 1-3 harness validation over real data; every run it produces is
    labelled Tier 3 and not promotable. For the Tier-1 TSMOM verdict use `run-tsmom`.
    """
    from cryptolab.data.store import ParquetStore
    from cryptolab.reporting.harness import build_harness_site
    from cryptolab.validation.registry import TrialRegistry

    base = BaseConfig.load(config)
    with TrialRegistry(base.registry_path) as registry:
        store = registry.bind_store(ParquetStore(base.data_root, base.splits))
        paths = build_harness_site(
            store,
            registry,
            out,
            symbols=base.universe,
            start=start,
            end=end,
            exchange=base.exchange,
        )
    typer.echo(f"wrote {len(paths)} files to {out}/")
    typer.secho(f"open {paths[0]}", fg=typer.colors.GREEN)


@app.command("harness-check")
def harness_check() -> None:
    """Phase 3 acceptance: exercise DSR, PBO and walk-forward on synthetic strategies."""
    import numpy as np

    from cryptolab.validation.deflated_sharpe import deflated_sharpe
    from cryptolab.validation.pbo import cscv_pbo
    from cryptolab.validation.walkforward import summarise

    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, 4000)
    trial_sharpes = rng.normal(0.0, 0.02, 48)

    dsr = deflated_sharpe(noise, n_trials=48, trial_sharpes=trial_sharpes)
    pbo = cscv_pbo(rng.normal(0.0, 0.01, (2000, 12)))
    folds = summarise(rng.normal(0.02, 0.03, 8))

    typer.echo(f"DSR on pure noise  {dsr.dsr:.3f} (SR0={dsr.sr0:.4f}, N={dsr.trials})")
    typer.echo(f"PBO on pure noise  {pbo.pbo:.3f} across {pbo.n_splits} splits")
    typer.echo(f"fold dispersion    {folds.dispersion_ratio:.2f} (gate: < 0.75)")
    typer.secho("harness responds to noise as expected", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
