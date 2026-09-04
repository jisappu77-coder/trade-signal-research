from __future__ import annotations

import json
import re

import polars as pl
import pytest

from cryptolab.backtest.costs import get_regime
from cryptolab.backtest.engine import BacktestConfig, run_backtest
from cryptolab.data.store import ParquetStore
from cryptolab.reporting import charts
from cryptolab.reporting.build import build_report, downsample
from cryptolab.reporting.harness import build_harness_site
from cryptolab.reporting.palette import (
    DARK,
    LIGHT,
    STATUS,
    VERDICT_STATUS,
    css_variables,
)
from cryptolab.reporting.report import SeriesPoint, StrategyReport, render_index, render_report, write_site
from cryptolab.validation.gates import GateInputs, evaluate_gates
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.synthetic import MomentumProbe, synthetic_bars

CONSERVATIVE = get_regime("conservative")


def make_gates(**overrides):
    base = {
        "net_sharpe_oos": 1.4,
        "deflated_sharpe": 0.98,
        "pbo": 0.12,
        "fold_sharpe_stdev": 0.4,
        "fold_sharpe_mean": 1.3,
        "max_drawdown": 0.22,
        "breakeven_cost_bps": 40.0,
        "parameter_plateau_fraction": 0.7,
        "beats_controls": True,
        "regime": CONSERVATIVE,
    }
    return evaluate_gates(GateInputs(**{**base, **overrides}))


def make_report(**overrides) -> StrategyReport:
    base = dict(
        strategy="TSMOM_L96_H72_1h",
        period="2024-07-01 → 2026-08-01",
        trials=48,
        regime=CONSERVATIVE,
        net_sharpe=1.4,
        deflated_sharpe=0.98,
        pbo=0.12,
        breakeven_bps=40.0,
        turnover=41.0,
        cost_drag_bps=451.0,
        max_drawdown=0.22,
        gates=make_gates(),
        equity=[SeriesPoint(f"2024-0{i + 1}-01", 100.0 + i, 101.0 + i) for i in range(6)],
        fold_sharpes=[1.2, 0.9, 1.4],
        regime_sharpes={"conservative": 1.4},
        grid_sharpes={"a": 1.4, "b": 1.1},
    )
    return StrategyReport(**{**base, **overrides})


# ---- the §12 contract ---------------------------------------------------------------


def test_header_block_has_every_mandatory_field_in_order():
    header = make_report().header_block()
    order = [
        "STRATEGY",
        "PERIOD",
        "TRIALS N",
        "COSTS",
        "NET SHARPE",
        "DEFLATED SHARPE",
        "PBO",
        "BREAKEVEN",
        "TURNOVER",
        "COST DRAG",
        "VERDICT",
    ]
    positions = [header.index(field) for field in order]
    assert positions == sorted(positions)


def test_verdict_comes_from_the_gates_not_the_template():
    """§12: the verdict is computed, never written by hand.

    Asserted by construction: the rendered verdict must equal what GateReport produces, so a
    template that hard-coded a friendlier wording would fail here.
    """
    report = make_report(gates=make_gates(pbo=0.9))
    computed = report.gates.verdict_line()
    assert report.verdict_line == computed
    assert "pbo" in computed
    assert computed.removeprefix("VERDICT    ") in render_report(report)


def test_a_failing_report_renders_and_names_its_failures():
    html = render_report(make_report(gates=make_gates(pbo=0.9, net_sharpe_oos=0.5)))
    assert "KILLED" in html or "FAIL" in html
    assert "pbo" in html


def test_report_is_self_contained():
    """No external stylesheet, script or image: a report must open from disk anywhere."""
    html = render_report(make_report())
    assert "<link" not in html
    assert not re.search(r'src\s*=\s*"https?://', html)
    assert "<style>" in html


def test_index_publishes_failures_alongside_passes():
    """§12: failed reports are retained and published — the index cannot hide them."""
    reports = [
        make_report(strategy="PASSES"),
        make_report(strategy="FAILS", gates=make_gates(pbo=0.9)),
        make_report(strategy="KILLED_ONE", gates=make_gates(net_sharpe_oos=0.1)),
    ]
    html = render_index(reports)
    for name in ("PASSES", "FAILS", "KILLED_ONE"):
        assert name in html
    assert "killed" in html


def test_index_counts_each_status():
    reports = [
        make_report(strategy="a"),
        make_report(strategy="b", gates=make_gates(pbo=0.9)),
        make_report(strategy="c", gates=make_gates(net_sharpe_oos=0.1)),
    ]
    html = render_index(reports)
    assert "Validated" in html and "Killed" in html


def test_index_says_so_when_nothing_has_passed():
    html = render_index([make_report(gates=make_gates(net_sharpe_oos=0.1))])
    assert "No strategy has passed" in html


def test_write_site_emits_index_report_and_json(tmp_path):
    paths = write_site([make_report()], tmp_path)
    assert paths[0].name == "index.html"
    assert (tmp_path / "tsmom_l96_h72_1h.html").exists()
    data = json.loads((tmp_path / "tsmom_l96_h72_1h.json").read_text())
    assert data["strategy"] == "TSMOM_L96_H72_1h" and data["trials"] == 48


# ---- the flat-book warning ----------------------------------------------------------


def test_a_mostly_flat_book_is_flagged_above_the_metrics():
    report = make_report(flat_fraction=0.96, kill_reason="max_gross_leverage", killed_from="2020-03-12")
    html = render_report(report)
    assert report.mostly_flat
    assert "BOOK WAS FLAT" in html and "2020-03-12" in html
    # The warning must precede the stat tiles it qualifies.
    assert html.index("BOOK WAS FLAT") < html.index("Net Sharpe")


def test_an_active_book_carries_no_flat_warning():
    html = render_report(make_report(flat_fraction=0.05))
    assert "BOOK WAS FLAT" not in html


def test_flat_fraction_reaches_the_index():
    html = render_index(
        [make_report(flat_fraction=0.96, kill_reason="max_gross_leverage", killed_from="2020-03-12")]
    )
    assert "flat" in html and "max_gross_leverage" in html


def test_non_promotable_runs_say_so():
    assert "not promotable" in render_report(make_report(promotable=False, tier=3))


# ---- charts -------------------------------------------------------------------------


def test_charts_emit_no_literal_colours():
    """Every mark reads a CSS token, so a theme swap repaints without new markup."""
    svg = (
        charts.line_chart([("net", [1.0, 2.0, 3.0], "series-1")], ["a", "b", "c"], label="t")
        + charts.diverging_bars(["a", "b"], [1.0, -1.0], label="t")
        + charts.ordinal_bars(["a"], [1.0], label="t")
    )
    assert "#" not in svg
    assert "var(--" in svg


def test_axis_ticks_are_distinct_over_a_wide_range():
    """The regression: a naive tick step collapsed every label to the same number."""
    ticks = charts._nice_ticks(25_000.0, 230_000.0)
    assert len(set(ticks)) == len(ticks) >= 2


def test_compact_formats_large_numbers_for_the_gutter():
    assert charts._compact(25_000) == "25k"
    assert charts._compact(1_200_000) == "1.2M"
    assert charts._compact(-0.75) == "-0.75"


def test_log_scale_is_used_only_for_positive_series():
    positive = charts.line_chart([("a", [1.0, 10.0, 100.0], "series-1")], list("abc"), label="t", log=True)
    negative = charts.line_chart([("a", [-1.0, 1.0], "series-1")], list("ab"), label="t", log=True)
    assert ">log<" in positive
    assert ">log<" not in negative  # a bankrupt path falls back to linear rather than crashing


def test_charts_survive_empty_input():
    assert "svg" in charts.line_chart([], [], label="t")
    assert "svg" in charts.diverging_bars([], [], label="t")


def test_every_chart_has_an_accessible_label():
    svg = charts.diverging_bars(["a"], [1.0], label="Fold Sharpe by fold")
    assert 'role="img"' in svg and 'aria-label="Fold Sharpe by fold"' in svg


def test_bars_carry_hover_titles():
    assert "<title>" in charts.diverging_bars(["fold 1"], [1.2], label="t")


# ---- palette ------------------------------------------------------------------------


def test_status_colours_are_never_reused_as_series_colours():
    status = set(STATUS.values())
    for tokens in (LIGHT, DARK):
        assert not status & {tokens["series-1"], tokens["series-2"]}


def test_theme_tokens_cover_both_modes_and_the_toggle():
    css = css_variables()
    assert "prefers-color-scheme: dark" in css
    assert '[data-theme="dark"]' in css
    assert ':not([data-theme="light"])' in css


def test_every_verdict_status_has_an_icon_and_a_word():
    """Colour never carries meaning alone."""
    for role, icon, word in VERDICT_STATUS.values():
        assert role in STATUS and icon and word


# ---- builder ------------------------------------------------------------------------


def test_downsample_keeps_the_final_bar(bars):
    targets = MomentumProbe().generate(bars, {"lookback": 24})
    result = run_backtest(bars, targets, BacktestConfig(warmup_bars=100))
    points = downsample(result, max_points=25)
    assert len(points) <= 25
    assert points[-1].net == pytest.approx(float(result.equity_curve["equity"][-1]))


def test_build_report_takes_its_numbers_from_the_result(bars):
    targets = MomentumProbe().generate(bars, {"lookback": 24})
    result = run_backtest(bars, targets, BacktestConfig(warmup_bars=100))
    report = build_report(
        strategy="probe",
        period="p",
        trials=8,
        result=result,
        gates=make_gates(),
        net_sharpe=0.5,
        deflated_sharpe=0.4,
        pbo=0.5,
    )
    assert report.breakeven_bps == result.breakeven_cost_bps()
    assert report.turnover == result.turnover_per_year
    assert report.flat_fraction == result.flat_fraction


def test_slug_is_filesystem_safe():
    assert make_report(strategy="TSMOM L96/H72 (1h)").slug == "tsmom-l96-h72--1h-"


# ---- the harness site builder -------------------------------------------------------


def test_harness_site_builds_from_a_store(tmp_path, splits):
    """End to end: ingest-shaped data in, a full site out, with trials registered."""
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    store = registry.bind_store(ParquetStore(tmp_path / "data", splits))

    # ~15 months of hourly bars from 2020-01-01, deep enough not to trip §7 capacity.
    frame = synthetic_bars(11_000, seed=5, start_ms=1_577_836_800_000, quote_volume=5e9)
    funding = pl.DataFrame(
        {
            "funding_time": frame["open_time"].to_list()[::8],
            "symbol": ["BTCUSDT"] * len(frame["open_time"].to_list()[::8]),
            "funding_rate": [0.0001] * len(frame["open_time"].to_list()[::8]),
            "interval_hours": [8.0] * len(frame["open_time"].to_list()[::8]),
            "mark_price": [20_000.0] * len(frame["open_time"].to_list()[::8]),
        }
    )
    for symbol in ("BTCUSDT", "ETHUSDT"):
        store.write(frame, "ohlcv", exchange="binance", symbol=symbol, source_uri="test://")
        store.write(
            funding.with_columns(pl.lit(symbol).alias("symbol")),
            "funding",
            exchange="binance",
            symbol=symbol,
            source_uri="test://",
        )

    paths = build_harness_site(store, registry, tmp_path / "site", start="2020-01-01", end="2021-03-01")

    assert paths[0].name == "index.html"
    assert len(paths) == 9  # index + 4 lookbacks x 2 symbols
    index = (tmp_path / "site" / "index.html").read_text()
    assert "tier 3" in index  # non-promotable runs are labelled in the index
    # Trials are registered per symbol (§8.1): 4 params x 2 symbols.
    assert registry.count(signal="momentum_probe") == 8
    assert "N = 8" in (tmp_path / "site" / "probe_l24_1h_btcusdt.html").read_text()
    registry.close()


def test_harness_site_refuses_an_empty_store(tmp_path, splits):
    registry = TrialRegistry(tmp_path / "r.sqlite")
    store = registry.bind_store(ParquetStore(tmp_path / "data", splits))
    with pytest.raises(ValueError, match="ingest the data first"):
        build_harness_site(store, registry, tmp_path / "site")
    registry.close()
