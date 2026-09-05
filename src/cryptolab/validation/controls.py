"""Controls for the §11 "beats controls" gate.

§11 requires a strategy to outperform BTC buy-and-hold, ETH buy-and-hold, **and a random-entry
control matched on exposure and turnover**. The first two are trivial; the third is the one that
actually matters, because it is what separates "this signal predicts something" from "this signal
takes the same risk, at the same trading intensity, and the market went up".

Matching is the whole point. An unmatched random control trades a different amount and holds a
different size, so beating it proves nothing. Here the control is calibrated to reproduce the
candidate's own exposure distribution and its realised turnover, then run over many seeds so the
comparison is against a *distribution* rather than a single lucky or unlucky draw.

Also here: the §11.1 single-quarter concentration criterion, which `GateInputs` has always accepted
and nothing ever computed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

DEFAULT_SEEDS = 40
# Turnover is matched by bisection on the flip probability; this is close enough that the remaining
# difference cannot flip a verdict.
TURNOVER_TOLERANCE = 0.05
MAX_CALIBRATION_STEPS = 24


@dataclass(frozen=True, slots=True)
class ControlResult:
    """The distribution of a control's out-of-sample performance."""

    name: str
    sharpes: tuple[float, ...]
    mean_sharpe: float
    p95_sharpe: float
    matched_turnover: float | None = None

    def beaten_by(self, candidate_sharpe: float) -> bool:
        """True when the candidate clears the control's 95th percentile.

        Compared against the tail, not the mean: a search that beats the *average* random strategy
        is not evidence, because the candidate was itself selected as a maximum over many trials.
        """
        return candidate_sharpe > self.p95_sharpe


def random_targets(n: int, exposure: np.ndarray, flip_probability: float, seed: int) -> np.ndarray:
    """A random target path matched to `exposure`'s size distribution.

    Direction is a random walk that flips with `flip_probability`; magnitude is drawn from the
    candidate's own |target| values, so the control holds the same sized positions as the strategy
    it is standing in for.
    """
    rng = np.random.default_rng(seed)
    sizes = np.abs(exposure[exposure != 0])
    if sizes.size == 0:
        return np.zeros(n)
    magnitude = rng.choice(sizes, size=n)
    flips = rng.random(n) < flip_probability
    direction = np.where(rng.random() < 0.5, -1.0, 1.0) * np.cumprod(np.where(flips, -1.0, 1.0))
    return direction * magnitude


def calibrate_flip_probability(
    target_turnover: float,
    measure: Callable[[float], float],
    *,
    tolerance: float = TURNOVER_TOLERANCE,
) -> float:
    """Bisect the flip probability until the control's turnover matches the candidate's.

    `measure` runs a backtest at a given flip probability and returns realised turnover.
    """
    if target_turnover <= 0:
        return 0.0
    low, high = 0.0, 1.0
    probability = 0.5
    for _ in range(MAX_CALIBRATION_STEPS):
        probability = (low + high) / 2
        achieved = measure(probability)
        if abs(achieved - target_turnover) <= tolerance * max(target_turnover, 1e-9):
            return probability
        if achieved < target_turnover:
            low = probability
        else:
            high = probability
    return probability


def buy_and_hold_sharpe(bars: pl.DataFrame, *, periods_per_year: float) -> float:
    """Annualised Sharpe of holding the asset — the simplest control, and often the hardest."""
    close = bars["close"].to_numpy()
    if close.size < 3:
        return 0.0
    rets = np.diff(close) / close[:-1]
    sd = float(np.std(rets, ddof=1))
    return 0.0 if sd == 0 else float(np.mean(rets)) / sd * float(np.sqrt(periods_per_year))


def summarise_controls(
    name: str, sharpes: Sequence[float], matched_turnover: float | None = None
) -> ControlResult:
    """Reduce many control runs to the distribution the gate compares against."""
    arr = np.asarray(list(sharpes), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ControlResult(name, (), 0.0, 0.0, matched_turnover)
    return ControlResult(
        name=name,
        sharpes=tuple(float(x) for x in arr),
        mean_sharpe=float(np.mean(arr)),
        p95_sharpe=float(np.percentile(arr, 95)),
        matched_turnover=matched_turnover,
    )


def single_quarter_concentrated(
    timestamps: Sequence[int], returns: Sequence[float], *, threshold: float = 0.8
) -> bool:
    """§11.1: is performance concentrated in a single calendar quarter?

    Declared in `GateInputs` since the gates were written and never computed until now, so this kill
    criterion silently defaulted to False for every run.

    True when one quarter accounts for more than `threshold` of the total positive P&L. A strategy
    whose entire edge is one quarter of one year has not shown an edge; it has shown one quarter.
    """
    times = np.asarray(list(timestamps), dtype=np.int64)
    rets = np.asarray(list(returns), dtype=float)
    if times.size == 0 or rets.size == 0:
        return False
    size = min(times.size, rets.size)
    times, rets = times[:size], rets[:size]

    def quarter_index(ms: int) -> int:
        moment = dt.datetime.fromtimestamp(ms / 1000, tz=dt.UTC)
        return moment.year * 4 + (moment.month - 1) // 3

    quarters = np.array([quarter_index(int(t)) for t in times])
    totals = {q: float(rets[quarters == q].sum()) for q in np.unique(quarters)}
    positive = [v for v in totals.values() if v > 0]
    if len(positive) < 2 or sum(positive) <= 0:
        return False
    return max(positive) / sum(positive) > threshold
