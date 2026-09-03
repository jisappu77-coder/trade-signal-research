from __future__ import annotations

import pytest

from cryptolab.data.store import to_ms
from cryptolab.validation.walkforward import (
    DAY_MS,
    generate_folds,
    summarise,
    walk_forward,
)

START, END = to_ms("2019-01-01"), to_ms("2024-06-30")


def test_anchored_folds_share_a_start():
    folds = generate_folds(START, END)
    assert len({f.train_start for f in folds}) == 1


def test_rolling_folds_have_constant_train_length():
    folds = generate_folds(START, END, mode="rolling")
    lengths = {f.train_end - f.train_start for f in folds}
    assert lengths == {365 * DAY_MS}


def test_folds_do_not_overlap_train_and_test():
    for fold in generate_folds(START, END):
        assert fold.test_start >= fold.train_end


def test_train_window_below_365_days_is_refused():
    with pytest.raises(ValueError, match="at least 365 days"):
        generate_folds(START, END, train_days=180)


def test_dispersion_ratio_is_the_headline_metric():
    """Mean OOS Sharpe 1.2 with fold stdev 1.4 is not a strategy (§10.2)."""
    summary = summarise([1.2, -0.9, 2.6, 1.1, -0.4, 3.0])
    assert summary.dispersion_ratio > 0.75 and not summary.passes


def test_a_consistent_strategy_passes_dispersion():
    summary = summarise([1.1, 1.2, 0.95, 1.05, 1.15])
    assert summary.passes


def test_negative_mean_scores_infinite_dispersion():
    assert summarise([-0.5, 0.2, -0.3]).dispersion_ratio == float("inf")


def test_single_fold_concentration_is_detected():
    """§11.1 kill criterion: all the performance sits in one fold."""
    assert summarise([0.05, 0.02, 4.0, 0.01, 0.03]).single_fold_concentrated


def test_evenly_spread_performance_is_not_flagged_as_concentrated():
    assert not summarise([1.0, 1.1, 0.9, 1.05]).single_fold_concentrated


def test_walk_forward_runs_every_fold():
    seen = []

    def evaluate(fold):
        seen.append(fold.index)
        return 1.0

    folds, summary = walk_forward(START, END, evaluate)
    assert seen == [f.index for f in folds]
    assert summary.mean_sharpe == 1.0


def test_a_window_too_short_for_one_fold_raises():
    with pytest.raises(ValueError, match="no folds fit"):
        walk_forward(START, START + 200 * DAY_MS, lambda _: 1.0)
