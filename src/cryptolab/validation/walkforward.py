"""Walk-forward validation (SPEC.md §10.2).

Anchored and rolling variants. **Fold dispersion is the headline robustness metric** — a strategy
with mean OOS Sharpe 1.2 and fold stdev 1.4 is not a strategy, and `FoldSummary.dispersion_ratio`
is what the §11 gate reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

DAY_MS = 86_400_000
Mode = Literal["anchored", "rolling"]


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, in UTC milliseconds."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class FoldSummary:
    """The distribution of OOS Sharpe across folds — reported instead of the aggregate."""

    fold_sharpes: tuple[float, ...]
    mean_sharpe: float
    stdev_sharpe: float

    @property
    def dispersion_ratio(self) -> float:
        """stdev / mean. The §11 gate requires < 0.75; a negative mean scores infinity."""
        if self.mean_sharpe <= 0:
            return float("inf")
        return self.stdev_sharpe / self.mean_sharpe

    @property
    def passes(self) -> bool:
        return self.dispersion_ratio < 0.75

    @property
    def single_fold_concentrated(self) -> bool:
        """§11.1 kill criterion: performance concentrated in one fold."""
        arr = np.asarray(self.fold_sharpes)
        if arr.size < 3 or arr.sum() <= 0:
            return False
        positive = arr[arr > 0]
        return positive.size > 0 and float(positive.max() / positive.sum()) > 0.8


def generate_folds(
    start_ms: int,
    end_ms: int,
    *,
    train_days: int = 365,
    test_days: int = 90,
    step_days: int = 90,
    mode: Mode = "anchored",
) -> list[Fold]:
    """Build the fold schedule. Train window >= 365d, test 90d, step 90d (§10.2 defaults)."""
    if train_days < 365:
        raise ValueError("train window must be at least 365 days (SPEC.md §10.2)")
    return list(
        _iter_folds(
            start_ms,
            end_ms,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            mode=mode,
        )
    )


def _iter_folds(
    start_ms: int,
    end_ms: int,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    mode: Mode,
) -> Iterator[Fold]:
    train_ms, test_ms, step_ms = train_days * DAY_MS, test_days * DAY_MS, step_days * DAY_MS
    index = 0
    train_end = start_ms + train_ms
    while train_end + test_ms <= end_ms:
        train_start = start_ms if mode == "anchored" else train_end - train_ms
        yield Fold(index, train_start, train_end, train_end, train_end + test_ms)
        index += 1
        train_end += step_ms


def summarise(fold_sharpes: list[float] | np.ndarray) -> FoldSummary:
    """Summarise fold Sharpes. All values must be in the same (per-observation) units."""
    arr = np.asarray(list(fold_sharpes), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return FoldSummary((), 0.0, 0.0)
    stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return FoldSummary(tuple(float(x) for x in arr), float(np.mean(arr)), stdev)


def walk_forward(
    start_ms: int,
    end_ms: int,
    evaluate: Callable[[Fold], float],
    *,
    train_days: int = 365,
    test_days: int = 90,
    step_days: int = 90,
    mode: Mode = "anchored",
) -> tuple[list[Fold], FoldSummary]:
    """Run `evaluate` on every fold and summarise the OOS Sharpe distribution.

    `evaluate` receives a fold and returns that fold's out-of-sample per-observation Sharpe.
    """
    folds = generate_folds(
        start_ms, end_ms, train_days=train_days, test_days=test_days, step_days=step_days, mode=mode
    )
    if not folds:
        raise ValueError(
            f"no folds fit in the window ({(end_ms - start_ms) / DAY_MS:.0f} days) for "
            f"train={train_days}d + test={test_days}d"
        )
    return folds, summarise([evaluate(f) for f in folds])
