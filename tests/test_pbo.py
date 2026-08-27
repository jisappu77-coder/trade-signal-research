from __future__ import annotations

import numpy as np
import pytest

from cryptolab.validation.pbo import cscv_pbo


def test_pure_noise_gives_pbo_near_one_half():
    """Selecting among noise configurations is a coin flip out of sample."""
    rng = np.random.default_rng(0)
    result = cscv_pbo(rng.normal(0.0, 0.01, (2000, 10)), n_splits=10)
    assert 0.3 < result.pbo < 0.7


def test_a_genuinely_dominant_configuration_gives_low_pbo():
    rng = np.random.default_rng(1)
    matrix = rng.normal(0.0, 0.01, (2000, 8))
    matrix[:, 3] += 0.004  # one config carries a real, persistent edge
    result = cscv_pbo(matrix, n_splits=10)
    assert result.pbo < 0.30 and result.passes


def test_needs_at_least_two_configurations():
    with pytest.raises(ValueError, match="at least 2 configurations"):
        cscv_pbo(np.zeros((100, 1)))


def test_splits_must_be_even():
    with pytest.raises(ValueError, match="must be even"):
        cscv_pbo(np.random.default_rng(2).normal(0, 1, (100, 4)), n_splits=7)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="must be 2-D"):
        cscv_pbo(np.zeros(100))


def test_split_count_is_the_balanced_combination_count():
    result = cscv_pbo(np.random.default_rng(3).normal(0, 0.01, (400, 4)), n_splits=8)
    assert result.n_splits == 70  # C(8, 4)
