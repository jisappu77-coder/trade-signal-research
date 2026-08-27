"""Promotion gates and the verdict line (SPEC.md §11, §12).

A strategy advances from `candidate` to `validated` only if **all** gates hold, under the
**conservative** cost regime, on out-of-sample data.

The verdict line is computed here and nowhere else. §12 requires it to be computed, never written
by hand, so `render_header` is the only sanctioned way to produce a report header.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cryptolab.backtest.costs import HEADLINE_REGIME, CostRegime

Status = Literal["validated", "candidate", "killed"]


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's outcome. `detail` is what the report prints when it fails."""

    name: str
    passed: bool
    value: float
    threshold: float
    detail: str


@dataclass(frozen=True, slots=True)
class GateInputs:
    """Everything the §11 table needs. All Sharpes are annualised — this is the report boundary."""

    net_sharpe_oos: float
    deflated_sharpe: float
    pbo: float
    fold_sharpe_stdev: float
    fold_sharpe_mean: float
    max_drawdown: float
    breakeven_cost_bps: float
    parameter_plateau_fraction: float
    beats_controls: bool
    regime: CostRegime
    single_fold_concentrated: bool = False
    single_quarter_concentrated: bool = False


@dataclass(slots=True)
class GateReport:
    """The full gate evaluation and the resulting status."""

    gates: list[GateResult] = field(default_factory=list)
    kill_reasons: list[str] = field(default_factory=list)
    inputs: GateInputs | None = None

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates) and not self.kill_reasons

    @property
    def status(self) -> Status:
        if self.kill_reasons:
            return "killed"
        return "validated" if self.passed else "candidate"

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def verdict_line(self) -> str:
        """The mandatory §12 verdict. Computed, never hand-written.

        Failures are named in the line itself — a report that says only "FAIL" tells the next
        reader nothing about which hurdle the edge died on.
        """
        if self.kill_reasons:
            return f"VERDICT    KILLED — {'; '.join(self.kill_reasons)} (SPEC.md §11.1: do not tune)"
        if self.passed:
            return "VERDICT    PASS — all §11 gates cleared under conservative costs"
        reasons = "; ".join(f"{g.name} {g.value:.2f} vs {g.threshold:.2f}" for g in self.failures)
        return f"VERDICT    FAIL — {reasons}"


def evaluate_gates(inputs: GateInputs) -> GateReport:
    """Evaluate every §11 gate. Costs must be the conservative regime — the headline number."""
    if inputs.regime.name != HEADLINE_REGIME:
        raise ValueError(
            f"gates are evaluated under the {HEADLINE_REGIME} regime only; got "
            f"{inputs.regime.name!r}. Evaluating gates under a cheaper regime is a protocol violation."
        )

    round_trip = inputs.regime.round_trip_taker_bps
    dispersion_threshold = 0.75 * inputs.fold_sharpe_mean

    gates = [
        GateResult(
            "net_sharpe_oos",
            inputs.net_sharpe_oos > 1.0,
            inputs.net_sharpe_oos,
            1.0,
            "annualised net Sharpe out of sample",
        ),
        GateResult(
            "deflated_sharpe",
            inputs.deflated_sharpe > 0.95,
            inputs.deflated_sharpe,
            0.95,
            "p < 0.05 after correcting for the trial count N from the registry",
        ),
        GateResult("pbo", inputs.pbo < 0.30, inputs.pbo, 0.30, "probability of backtest overfitting"),
        GateResult(
            "fold_dispersion",
            inputs.fold_sharpe_mean > 0 and inputs.fold_sharpe_stdev < dispersion_threshold,
            inputs.fold_sharpe_stdev,
            dispersion_threshold,
            "walk-forward fold Sharpe stdev < 0.75 x mean fold Sharpe",
        ),
        GateResult(
            "max_drawdown", inputs.max_drawdown < 0.35, inputs.max_drawdown, 0.35, "peak-to-trough"
        ),
        GateResult(
            "breakeven_cost",
            inputs.breakeven_cost_bps > 2.0 * round_trip,
            inputs.breakeven_cost_bps,
            2.0 * round_trip,
            f"break-even must exceed 2x the conservative round-trip ({round_trip:.1f} bps)",
        ),
        GateResult(
            "parameter_plateau",
            inputs.parameter_plateau_fraction >= 0.50,
            inputs.parameter_plateau_fraction,
            0.50,
            "Sharpe within 25% of peak across >=50% of neighbouring grid",
        ),
        GateResult(
            "beats_controls",
            inputs.beats_controls,
            1.0 if inputs.beats_controls else 0.0,
            1.0,
            "vs BTC buy-and-hold, ETH buy-and-hold, and an exposure/turnover-matched random control",
        ),
    ]

    kills: list[str] = []
    if inputs.net_sharpe_oos < 0.3:
        kills.append(f"net Sharpe {inputs.net_sharpe_oos:.2f} < 0.3 under conservative costs")
    if inputs.breakeven_cost_bps < round_trip:
        kills.append(
            f"break-even {inputs.breakeven_cost_bps:.1f} bps below the expected round-trip "
            f"{round_trip:.1f} bps"
        )
    if inputs.single_fold_concentrated:
        kills.append("performance concentrated in a single fold")
    if inputs.single_quarter_concentrated:
        kills.append("performance concentrated in a single quarter")

    return GateReport(gates=gates, kill_reasons=kills, inputs=inputs)


def parameter_plateau_fraction(sharpes: dict[str, float]) -> float:
    """Fraction of the grid scoring within 25% of the peak Sharpe (§11 plateau gate).

    A lone spike in an otherwise flat grid is a fitted artifact, not a plateau.
    """
    if not sharpes:
        return 0.0
    values = list(sharpes.values())
    peak = max(values)
    if peak <= 0:
        return 0.0
    return sum(1 for v in values if v >= 0.75 * peak) / len(values)


def render_header(
    *,
    strategy: str,
    period: str,
    trials: int,
    regime: CostRegime,
    net_sharpe: float,
    deflated: float,
    pbo: float,
    breakeven_bps: float,
    turnover: float,
    cost_drag_bps: float,
    report: GateReport,
) -> str:
    """Render the mandatory §12 header block, verdict line included."""
    return "\n".join(
        [
            f"STRATEGY   {strategy:<30}PERIOD  {period}",
            f"TRIALS N   {trials:<30}COSTS   {regime.name} ({regime.taker_fee_bps}bps taker)",
            f"NET SHARPE {net_sharpe:<8.2f}DEFLATED SHARPE {deflated:<6.2f}PBO     {pbo:.2f}",
            f"BREAKEVEN  {breakeven_bps:.1f} bps round-trip"
            f"{'':<12}TURNOVER {turnover:.0f}x/yr   COST DRAG {cost_drag_bps:.0f} bps/yr",
            report.verdict_line(),
        ]
    )
