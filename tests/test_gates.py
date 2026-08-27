from __future__ import annotations

import pytest

from cryptolab.backtest.costs import get_regime
from cryptolab.validation.gates import (
    GateInputs,
    evaluate_gates,
    parameter_plateau_fraction,
    render_header,
)

CONSERVATIVE = get_regime("conservative")


def passing_inputs(**overrides) -> GateInputs:
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
    return GateInputs(**{**base, **overrides})


def test_a_fully_passing_strategy_is_validated():
    report = evaluate_gates(passing_inputs())
    assert report.status == "validated"
    assert report.verdict_line().startswith("VERDICT    PASS")


@pytest.mark.parametrize(
    ("override", "gate"),
    [
        ({"net_sharpe_oos": 0.9}, "net_sharpe_oos"),
        ({"deflated_sharpe": 0.5}, "deflated_sharpe"),
        ({"pbo": 0.41}, "pbo"),
        ({"fold_sharpe_stdev": 1.4}, "fold_dispersion"),
        ({"max_drawdown": 0.4}, "max_drawdown"),
        # 20 bps clears the 15 bps round-trip (so no §11.1 kill) but misses the 2x gate.
        ({"breakeven_cost_bps": 20.0}, "breakeven_cost"),
        ({"parameter_plateau_fraction": 0.2}, "parameter_plateau"),
        ({"beats_controls": False}, "beats_controls"),
    ],
)
def test_every_gate_can_fail_alone(override, gate):
    report = evaluate_gates(passing_inputs(**override))
    assert not report.passed
    assert [g.name for g in report.failures] == [gate]
    assert gate in report.verdict_line()


def test_the_spec_worked_example_fails():
    """The §12 header example: net 0.84, deflated 0.31, PBO 0.41 — a FAIL, and it must read as one."""
    report = evaluate_gates(
        passing_inputs(net_sharpe_oos=0.84, deflated_sharpe=0.31, pbo=0.41, breakeven_cost_bps=6.2)
    )
    assert report.status == "killed"  # break-even 6.2 bps is below the round-trip
    assert "KILLED" in report.verdict_line()


def test_kill_criteria_take_precedence_over_gate_failures():
    report = evaluate_gates(passing_inputs(net_sharpe_oos=0.1))
    assert report.status == "killed"
    assert "do not tune" in report.verdict_line()


def test_break_even_below_round_trip_is_a_kill():
    report = evaluate_gates(passing_inputs(breakeven_cost_bps=CONSERVATIVE.round_trip_taker_bps - 1))
    assert report.status == "killed"


def test_single_fold_concentration_is_a_kill():
    assert evaluate_gates(passing_inputs(single_fold_concentrated=True)).status == "killed"


def test_gates_refuse_a_cheaper_regime():
    """Evaluating gates under anything but conservative is a protocol violation."""
    for name in ("optimistic", "expected", "stressed"):
        with pytest.raises(ValueError, match="conservative regime only"):
            evaluate_gates(passing_inputs(regime=get_regime(name)))


def test_break_even_gate_is_two_round_trips():
    round_trip = CONSERVATIVE.round_trip_taker_bps
    assert evaluate_gates(passing_inputs(breakeven_cost_bps=2 * round_trip + 0.1)).passed
    assert not evaluate_gates(passing_inputs(breakeven_cost_bps=2 * round_trip - 0.1)).passed


def test_verdict_line_names_every_failing_gate():
    report = evaluate_gates(passing_inputs(pbo=0.5, max_drawdown=0.5))
    line = report.verdict_line()
    assert "pbo" in line and "max_drawdown" in line


def test_plateau_fraction_rewards_a_flat_grid():
    flat = dict.fromkeys("abcdefgh", 1.0)
    assert parameter_plateau_fraction(flat) == 1.0


def test_plateau_fraction_punishes_a_lone_spike():
    spike = {"a": 2.0, **{k: 0.1 for k in "bcdefgh"}}
    assert parameter_plateau_fraction(spike) < 0.5


def test_plateau_fraction_of_an_all_negative_grid_is_zero():
    assert parameter_plateau_fraction({"a": -1.0, "b": -2.0}) == 0.0


def test_header_contains_every_mandatory_field():
    report = evaluate_gates(passing_inputs(net_sharpe_oos=0.84, deflated_sharpe=0.31, pbo=0.41))
    header = render_header(
        strategy="TSMOM_L96_H72_1h",
        period="2024-07-01 → 2026-08-01  (OOS, sealed)",
        trials=48,
        regime=CONSERVATIVE,
        net_sharpe=0.84,
        deflated=0.31,
        pbo=0.41,
        breakeven_bps=6.2,
        turnover=41,
        cost_drag_bps=451,
        report=report,
    )
    for field in ("STRATEGY", "TRIALS N", "NET SHARPE", "DEFLATED SHARPE", "PBO", "BREAKEVEN",
                  "TURNOVER", "COST DRAG", "VERDICT"):
        assert field in header
    assert "48" in header and "conservative" in header
