"""Deflated Sharpe ratio, Bailey & López de Prado (2014) — SPEC.md §10.3.

    SR0 = stdev(SR_trials) * [ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)) ]
    DSR = Φ( (SR_hat - SR0)·√(T-1) / √(1 - γ3·SR_hat + ((γ4-1)/4)·SR_hat²) )

**Units.** `SR_hat`, `SR0` and `stdev(SR_trials)` are all per-observation Sharpes at the same
frequency as the `T` observations. This module refuses annualised inputs (see `_assert_not_annualised`)
because annualising one side and not the other silently inflates DSR — the single easiest way to
report a false positive here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

EULER_MASCHERONI: Final[float] = 0.5772156649015329

# A per-observation Sharpe above this is almost certainly an annualised number passed by mistake.
_SANE_PER_OBS_SHARPE: Final[float] = 1.5


class SharpeUnitsError(ValueError):
    """Raised when a Sharpe looks annualised but a per-observation value is required."""


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """The full DSR computation, kept together so a report can never show SR without its DSR."""

    sharpe_per_obs: float
    sharpe_annualised: float
    dsr: float
    sr0: float
    trials: int
    observations: int
    skew: float
    kurtosis: float

    @property
    def passes(self) -> bool:
        """§11 gate: DSR > 0.95, i.e. p < 0.05 after trial correction."""
        return self.dsr > 0.95


def _assert_not_annualised(value: float, name: str) -> None:
    if abs(value) > _SANE_PER_OBS_SHARPE:
        raise SharpeUnitsError(
            f"{name}={value:.3f} looks annualised. This function takes per-observation Sharpes "
            "(SPEC.md §10.3); annualise only at the reporting boundary."
        )


def expected_max_sharpe(trial_sharpe_stdev: float, n_trials: int) -> float:
    """`SR0` — the Sharpe you expect from the best of `n_trials` pure-noise strategies.

    This is the benchmark a real edge must beat. With N=48 and trial dispersion of 0.05 per bar, a
    noise-only search is *expected* to turn up a Sharpe well above zero — which is the entire point.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if trial_sharpe_stdev < 0:
        raise ValueError("trial_sharpe_stdev must be non-negative")
    if n_trials == 1:
        return 0.0
    gamma = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sharpe_stdev * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe(
    returns: np.ndarray,
    *,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    trial_sharpe_stdev: float | None = None,
    periods_per_year: float = 8766.0,
) -> DeflatedSharpeResult:
    """Compute DSR for a return series.

    Either `trial_sharpes` (the per-observation Sharpes of every registered trial) or an explicit
    `trial_sharpe_stdev` must be supplied — `SR0` is meaningless without the dispersion of the
    search that produced the candidate.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t_obs = r.size
    if t_obs < 3:
        raise ValueError(f"need at least 3 return observations, got {t_obs}")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1; read it from the trial registry, never hard-code it")

    sd = float(np.std(r, ddof=1))
    sr_hat = 0.0 if sd == 0 else float(np.mean(r)) / sd
    _assert_not_annualised(sr_hat, "sharpe_per_obs")

    if trial_sharpe_stdev is None:
        if trial_sharpes is None:
            raise ValueError("supply trial_sharpes or trial_sharpe_stdev — SR0 needs trial dispersion")
        arr = np.asarray(trial_sharpes, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            raise ValueError("need >= 2 trial Sharpes to estimate dispersion")
        for value in (float(np.max(np.abs(arr))),):
            _assert_not_annualised(value, "trial_sharpes")
        trial_sharpe_stdev = float(np.std(arr, ddof=1))

    sr0 = expected_max_sharpe(trial_sharpe_stdev, n_trials)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    denominator_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if denominator_sq <= 0:
        # Extreme higher moments; the variance estimate of SR breaks down. Report DSR 0 rather
        # than a complex number, and let the gate fail honestly.
        dsr = 0.0
    else:
        z = (sr_hat - sr0) * math.sqrt(t_obs - 1) / math.sqrt(denominator_sq)
        dsr = float(stats.norm.cdf(z))

    return DeflatedSharpeResult(
        sharpe_per_obs=sr_hat,
        sharpe_annualised=sr_hat * math.sqrt(periods_per_year),
        dsr=dsr,
        sr0=sr0,
        trials=n_trials,
        observations=t_obs,
        skew=skew,
        kurtosis=kurt,
    )
