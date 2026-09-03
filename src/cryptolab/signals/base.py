"""The Signal interface (SPEC.md §8).

`generate` MUST be causal: row i may reference feature values at rows <= i only. The shift test
(§14.2.1) enforces this mechanically for every signal, and it runs in CI.

`target_position` is a *desired exposure* in [-1, 1], not an order. The portfolio layer converts
desire into orders and applies the no-trade band (§9.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, ClassVar, Literal

import polars as pl

Tier = Literal[1, 2, 3]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A feature dependency with its lookback declared in bars.

    The declared lookback is what the engine trims from the front of every backtest window, so an
    understated lookback shows up as a warm-up artifact rather than silent lookahead.
    """

    name: str
    lookback_bars: int


@dataclass(frozen=True, slots=True)
class ParamRange:
    """A declared parameter axis. The product of all axes is the trial count (§8.1)."""

    name: str
    values: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.values)


class Signal(ABC):
    """Base class for every signal family."""

    name: str
    tier: Tier
    # Class-level contracts: a signal's feature dependencies and its declared search space are
    # properties of the family, not of an instance. Trial counting depends on that being fixed.
    required_features: ClassVar[list[FeatureSpec]]
    param_space: ClassVar[dict[str, ParamRange]]

    @abstractmethod
    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        """Return columns: timestamp, target_position (float in [-1, 1]), confidence.

        MUST be causal. Row i may only reference feature values at rows <= i. The engine enforces
        this with the shift test in §14.2.
        """

    @property
    def max_lookback_bars(self) -> int:
        """Longest declared lookback — the warm-up the engine must discard."""
        return max((f.lookback_bars for f in self.required_features), default=0)

    def grid(self) -> list[dict[str, Any]]:
        """Every declared parameter combination, in a stable order.

        This is the *entire* declared search space. Expanding it is a registrable event that raises
        `N` and lowers the deflated Sharpe accordingly (CLAUDE.md, §8.1).
        """
        if not self.param_space:
            return [{}]
        names = sorted(self.param_space)
        axes = [self.param_space[n].values for n in names]
        return [dict(zip(names, combo, strict=True)) for combo in product(*axes)]

    def validate_output(self, df: pl.DataFrame) -> pl.DataFrame:
        """Assert the output contract: required columns, dtypes, and target_position in [-1, 1]."""
        required = {"timestamp": pl.Int64, "target_position": pl.Float64, "confidence": pl.Float64}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}.generate is missing columns {missing}")
        out = df.select(
            pl.col("timestamp").cast(pl.Int64),
            pl.col("target_position").cast(pl.Float64),
            pl.col("confidence").cast(pl.Float64),
        )
        if out["target_position"].is_null().any():
            raise ValueError(f"{self.name}.generate produced null target_position")
        # NaN is caught by the range check below only as an accident of polars' comparison
        # semantics, and reported as "outside [-1, 1]", which sends the reader hunting for a
        # leverage bug. A vol-scaled signal reaches NaN via 0 * inf when sigma is zero.
        if out["target_position"].is_nan().any():
            raise ValueError(
                f"{self.name}.generate produced NaN target_position; a vol-scaled signal reaches "
                "this when sigma is zero — fill it before returning"
            )
        extreme = out.filter(pl.col("target_position").abs() > 1.0 + 1e-9)
        if extreme.height:
            raise ValueError(
                f"{self.name}.generate produced target_position outside [-1, 1] "
                f"({extreme.height} rows); leverage belongs in the portfolio layer"
            )
        if not out["timestamp"].is_sorted():
            raise ValueError(f"{self.name}.generate produced unsorted timestamps")
        return out
