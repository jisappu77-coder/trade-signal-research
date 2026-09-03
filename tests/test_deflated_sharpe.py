from __future__ import annotations

import numpy as np
import pytest

from cryptolab.validation.deflated_sharpe import (
    SharpeUnitsError,
    deflated_sharpe,
    expected_max_sharpe,
)


def test_sr0_is_zero_for_a_single_trial():
    assert expected_max_sharpe(0.05, 1) == 0.0


def test_sr0_grows_with_trial_count():
    values = [expected_max_sharpe(0.05, n) for n in (2, 10, 48, 400)]
    assert values == sorted(values)


def test_sr0_grows_with_trial_dispersion():
    assert expected_max_sharpe(0.10, 48) > expected_max_sharpe(0.05, 48)


def test_pure_noise_does_not_clear_the_gate():
    """The whole point: a good-looking Sharpe drawn from a 48-trial search must not pass."""
    rng = np.random.default_rng(0)
    result = deflated_sharpe(
        rng.normal(0.0, 0.01, 4000), n_trials=48, trial_sharpes=rng.normal(0.0, 0.02, 48)
    )
    assert not result.passes


def test_a_strong_genuine_edge_does_clear_the_gate():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0015, 0.01, 6000)  # ~0.15 Sharpe per bar
    result = deflated_sharpe(returns, n_trials=48, trial_sharpe_stdev=0.02)
    assert result.passes and result.dsr > 0.99


def test_more_trials_lower_the_dsr():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0006, 0.01, 4000)
    few = deflated_sharpe(returns, n_trials=2, trial_sharpe_stdev=0.03)
    many = deflated_sharpe(returns, n_trials=400, trial_sharpe_stdev=0.03)
    assert many.dsr < few.dsr
    assert few.sharpe_per_obs == pytest.approx(many.sharpe_per_obs)


def test_annualised_sharpe_input_is_refused():
    """The §10.3 units trap: an annualised SR must not be silently accepted."""
    rng = np.random.default_rng(3)
    with pytest.raises(SharpeUnitsError, match="looks annualised"):
        deflated_sharpe(rng.normal(0.0, 0.01, 1000), n_trials=10, trial_sharpes=np.full(10, 2.5))


def test_returns_scaled_to_annual_units_are_refused():
    annualised_returns = np.random.default_rng(4).normal(0.9, 0.1, 500)
    with pytest.raises(SharpeUnitsError):
        deflated_sharpe(annualised_returns, n_trials=10, trial_sharpe_stdev=0.02)


def test_dispersion_is_mandatory():
    with pytest.raises(ValueError, match="trial dispersion"):
        deflated_sharpe(np.random.default_rng(5).normal(0, 0.01, 500), n_trials=10)


def test_trial_count_must_be_positive():
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        deflated_sharpe(np.random.default_rng(6).normal(0, 0.01, 500), n_trials=0, trial_sharpe_stdev=0.01)


def test_annualisation_is_reporting_only():
    rng = np.random.default_rng(7)
    result = deflated_sharpe(
        rng.normal(0.0003, 0.01, 3000), n_trials=24, trial_sharpe_stdev=0.02, periods_per_year=8766.0
    )
    assert result.sharpe_annualised == pytest.approx(result.sharpe_per_obs * np.sqrt(8766.0))
