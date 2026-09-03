"""Probability of backtest overfitting via CSCV (SPEC.md §10.5).

Bailey, Borwein, López de Prado & Zhu (2015). The performance matrix is split into `S` disjoint
submatrices; for every balanced combination of half-in-sample / half-out-of-sample, the IS-best
configuration is located and its OOS rank recorded. PBO is the fraction of splits where that
IS-winner lands in the bottom half out of sample — i.e. how often the selection procedure itself
is picking noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True, slots=True)
class PBOResult:
    """PBO plus the logit distribution it is computed from."""

    pbo: float
    n_splits: int
    n_configs: int
    logits: tuple[float, ...]
    median_oos_rank: float

    @property
    def passes(self) -> bool:
        """§11 gate: PBO < 0.30."""
        return self.pbo < 0.30


def cscv_pbo(performance: np.ndarray, n_splits: int = 16) -> PBOResult:
    """Compute PBO from a (T observations x C configurations) matrix of returns.

    `n_splits` must be even; the number of balanced combinations is C(S, S/2), so S=16 gives 12870
    splits — the standard choice, and enough that the estimate is stable.
    """
    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"performance must be 2-D (observations x configs), got shape {matrix.shape}")
    n_obs, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least 2 configurations to choose between")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so the splits are balanced")
    if n_obs < n_splits:
        raise ValueError(f"need at least {n_splits} observations to form {n_splits} submatrices")

    # Trim to a multiple of n_splits so the submatrices are equal length.
    usable = n_obs - (n_obs % n_splits)
    blocks = np.array_split(matrix[:usable], n_splits, axis=0)
    half = n_splits // 2

    logits: list[float] = []
    ranks: list[float] = []
    for is_idx in combinations(range(n_splits), half):
        oos_idx = tuple(i for i in range(n_splits) if i not in is_idx)
        is_matrix = np.vstack([blocks[i] for i in is_idx])
        oos_matrix = np.vstack([blocks[i] for i in oos_idx])

        best = int(np.argmax(_sharpes(is_matrix)))
        oos_sharpes = _sharpes(oos_matrix)
        # Relative rank of the IS winner among OOS results, in (0, 1].
        rank = float((np.sum(oos_sharpes <= oos_sharpes[best])) / n_configs)
        rank = min(max(rank, 1.0 / (n_configs + 1)), n_configs / (n_configs + 1))
        ranks.append(rank)
        logits.append(float(np.log(rank / (1.0 - rank))))

    logit_array = np.asarray(logits)
    return PBOResult(
        pbo=float(np.mean(logit_array <= 0.0)),
        n_splits=len(logits),
        n_configs=n_configs,
        logits=tuple(logits),
        median_oos_rank=float(np.median(ranks)),
    )


def _sharpes(matrix: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe of each column; zero-variance columns score zero, not inf."""
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    out: np.ndarray = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    return out
