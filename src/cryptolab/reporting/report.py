"""Static HTML reports and the comparison index (SPEC.md §12).

One report per strategy run; one index over all of them. **Failed reports are retained and
published** — §12 calls the record of what did not work the most valuable artifact this system
produces, so the index shows failures with the same prominence as passes, and nothing here can
filter them out.

The verdict line is never written by hand: it comes from `validation.gates.GateReport.verdict_line`.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from cryptolab.backtest.costs import CostRegime
from cryptolab.reporting import charts
from cryptolab.reporting.palette import VERDICT_STATUS, css_variables
from cryptolab.validation.gates import GateReport, render_header

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    """One sample of the equity path, already downsampled for rendering."""

    label: str
    net: float
    gross: float


@dataclass(slots=True)
class StrategyReport:
    """Everything one report renders. Assembled by the caller from a backtest plus validation."""

    strategy: str
    period: str
    trials: int
    regime: CostRegime
    net_sharpe: float
    deflated_sharpe: float
    pbo: float
    breakeven_bps: float
    turnover: float
    cost_drag_bps: float
    max_drawdown: float
    gates: GateReport
    flat_fraction: float = 0.0
    kill_reason: str | None = None
    killed_from: str | None = None
    equity: list[SeriesPoint] = field(default_factory=list)
    fold_sharpes: list[float] = field(default_factory=list)
    regime_sharpes: dict[str, float] = field(default_factory=dict)
    grid_sharpes: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tier: int = 1
    promotable: bool = True
    generated_at: str = ""

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "-" for c in self.strategy).lower()

    @property
    def mostly_flat(self) -> bool:
        """True when the book held no position for most of the period.

        Metrics over such a window are computed largely over cash and cannot be read as ordinary
        performance, so the report says so above the numbers rather than below them.
        """
        return self.flat_fraction > 0.5

    @property
    def status(self) -> str:
        return self.gates.status

    @property
    def verdict_line(self) -> str:
        """§12: mandatory, and computed rather than written."""
        return self.gates.verdict_line()

    def header_block(self) -> str:
        """The §12 header, rendered in the required order as monospace text."""
        return render_header(
            strategy=self.strategy,
            period=self.period,
            trials=self.trials,
            regime=self.regime,
            net_sharpe=self.net_sharpe,
            deflated=self.deflated_sharpe,
            pbo=self.pbo,
            breakeven_bps=self.breakeven_bps,
            turnover=self.turnover,
            cost_drag_bps=self.cost_drag_bps,
            report=self.gates,
        )

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable summary, written beside each report so the index can be rebuilt."""
        return {
            "strategy": self.strategy,
            "slug": self.slug,
            "period": self.period,
            "trials": self.trials,
            "regime": self.regime.name,
            "net_sharpe": self.net_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
            "breakeven_bps": self.breakeven_bps,
            "turnover": self.turnover,
            "cost_drag_bps": self.cost_drag_bps,
            "max_drawdown": self.max_drawdown,
            "flat_fraction": self.flat_fraction,
            "kill_reason": self.kill_reason,
            "killed_from": self.killed_from,
            "status": self.status,
            "verdict": self.verdict_line,
            "tier": self.tier,
            "promotable": self.promotable,
            "generated_at": self.generated_at,
            "gates": [asdict(g) for g in self.gates.gates],
            "kill_reasons": self.gates.kill_reasons,
        }


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["num"] = lambda v, d=2: f"{v:,.{d}f}"
    env.filters["pct"] = lambda v: f"{v:.1%}"
    return env


def _charts_for(report: StrategyReport) -> dict[str, str]:
    """Build every figure for one report. Empty inputs yield an empty string, not a broken axis."""
    out: dict[str, str] = {}
    if report.equity:
        out["equity"] = charts.line_chart(
            [
                ("net", [p.net for p in report.equity], "series-1"),
                ("gross", [p.gross for p in report.equity], "series-2"),
            ],
            [p.label for p in report.equity],
            label=f"Equity curve for {report.strategy}, net and gross of costs",
            log=True,
        )
    if report.fold_sharpes:
        out["folds"] = charts.diverging_bars(
            [f"fold {i + 1}" for i in range(len(report.fold_sharpes))],
            report.fold_sharpes,
            label="Out-of-sample Sharpe by walk-forward fold",
            reference=1.0,
        )
    if report.regime_sharpes:
        out["regimes"] = charts.ordinal_bars(
            list(report.regime_sharpes),
            list(report.regime_sharpes.values()),
            label="Net Sharpe under each cost regime",
            highlight=report.regime.name,
        )
    if report.grid_sharpes:
        out["grid"] = charts.diverging_bars(
            list(report.grid_sharpes),
            list(report.grid_sharpes.values()),
            label="Net Sharpe across the declared parameter grid",
        )
    return out


def render_report(report: StrategyReport) -> str:
    """Render one strategy report to a self-contained HTML string."""
    env = _environment()
    role, icon, verdict_word = VERDICT_STATUS[report.status]
    return env.get_template("report.html.j2").render(
        r=report,
        charts=_charts_for(report),
        css_vars=css_variables(),
        status_role=role,
        status_icon=icon,
        verdict_word=verdict_word,
        gate_bar=charts.gate_bar,
    )


def render_index(reports: Sequence[StrategyReport]) -> str:
    """Render the comparison index over every run, failures included."""
    env = _environment()
    ordered = sorted(reports, key=lambda r: (r.status != "validated", -r.net_sharpe))
    counts = {
        "validated": sum(1 for r in reports if r.status == "validated"),
        "candidate": sum(1 for r in reports if r.status == "candidate"),
        "killed": sum(1 for r in reports if r.status == "killed"),
    }
    return env.get_template("index.html.j2").render(
        reports=ordered,
        counts=counts,
        total=len(reports),
        css_vars=css_variables(),
        verdict_status=VERDICT_STATUS,
        generated_at=dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def write_site(reports: Sequence[StrategyReport], out_dir: Path | str) -> list[Path]:
    """Write the index and every report. Returns the paths written, index first."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "index.html"]
    written[0].write_text(render_index(reports), encoding="utf-8")
    for report in reports:
        path = out / f"{report.slug}.html"
        path.write_text(render_report(report), encoding="utf-8")
        (out / f"{report.slug}.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        written.append(path)
    return written
