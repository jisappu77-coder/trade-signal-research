"""Feature registry: name -> callable, with a declared lookback (SPEC.md §4).

The declared lookback is not documentation. The engine trims that many bars off the front of every
window, so an understated lookback shows up as a warm-up artifact instead of silent lookahead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

ExprFactory = Callable[..., pl.Expr]


@dataclass(frozen=True, slots=True)
class RegisteredFeature:
    """A feature and the lookback it needs, in bars."""

    name: str
    factory: ExprFactory
    lookback_bars: int
    description: str = ""


class FeatureRegistry:
    """A small explicit registry — no import-time side effects, no global mutable state."""

    def __init__(self) -> None:
        self._features: dict[str, RegisteredFeature] = {}

    def register(
        self, name: str, factory: ExprFactory, *, lookback_bars: int, description: str = ""
    ) -> RegisteredFeature:
        if name in self._features:
            raise KeyError(f"feature {name!r} is already registered")
        feature = RegisteredFeature(name, factory, lookback_bars, description)
        self._features[name] = feature
        return feature

    def get(self, name: str) -> RegisteredFeature:
        if name not in self._features:
            raise KeyError(f"unknown feature {name!r}; registered: {sorted(self._features)}")
        return self._features[name]

    def names(self) -> list[str]:
        return sorted(self._features)

    def max_lookback(self, names: list[str]) -> int:
        """The warm-up any frame using `names` must discard."""
        return max((self.get(n).lookback_bars for n in names), default=0)

    def build(self, df: pl.DataFrame, specs: dict[str, dict[str, Any]]) -> pl.DataFrame:
        """Apply the named features with their kwargs, in a deterministic order."""
        exprs = [self.get(name).factory(**kwargs) for name, kwargs in sorted(specs.items())]
        return df.with_columns(exprs)


def default_registry() -> FeatureRegistry:
    """The standard feature set. Constructed on call — never at import time."""
    # Deferred so importing the registry has no side effects (CLAUDE.md: no module-level I/O).
    from cryptolab.features import derivatives, returns, volatility  # noqa: PLC0415

    registry = FeatureRegistry()
    registry.register("log_return", returns.log_returns, lookback_bars=1, description="log(c_t/c_t-1)")
    registry.register("momentum", returns.momentum, lookback_bars=168, description="TSMOM raw input")
    registry.register("ewm_stdev", volatility.ewm_stdev, lookback_bars=144, description="EWM vol")
    registry.register(
        "realised_vol", volatility.realised_vol, lookback_bars=180 * 6, description="regime input"
    )
    registry.register(
        "drawdown_from_high",
        volatility.drawdown_from_high,
        lookback_bars=60 * 6,
        description="regime input",
    )
    registry.register("funding_apr", derivatives.funding_apr, lookback_bars=1, description="carry input")
    registry.register("basis_bps", derivatives.basis_bps, lookback_bars=1, description="mark vs index")
    return registry
