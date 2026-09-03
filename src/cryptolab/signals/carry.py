"""CARRY — funding cash-and-carry (SPEC.md §8.3).

**Not directional.** This is a delta-neutral yield sleeve: long spot, short perp, collecting funding
while it is positive enough to clear the cost of holding both legs. It needs no directional
accuracy at all, which is what distinguishes it from every other family here — the
[horizon analysis](../../docs/horizon-analysis.md) shows directional intraday trading needs 61-92%
accuracy to break even, and carry sidesteps that question entirely.

    funding_apr     = funding_rate * (8760 / funding_interval_hours)
    entry_threshold = 2 * round_trip_cost_apr_equivalent + margin_buffer

`target_position` here means **sleeve deployment in [0, 1]**, not directional exposure. The delta
hedge is structural — both legs are always the same size — so the position carries no price view.

Entry and exit use hysteresis: enter above `entry_threshold`, hold until funding falls below
`exit_threshold`. Without it, funding oscillating around a single level would churn both legs and
pay four fills for nothing, which is the failure mode §8.3's funding-flip requirement exists to
catch.

§8.3's empirical prior is that only ~40% of apparently attractive opportunities remain profitable
after costs and spread reversals. `backtest.carry` reports the realised hit rate against it.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import numpy as np
import polars as pl

from cryptolab.signals.base import FeatureSpec, ParamRange, Signal

HOURS_PER_YEAR: Final[float] = 8760.0

# §8.3 defaults. `margin_buffer` is the extra APR demanded above pure cost recovery — the premium
# for tying up capital and carrying liquidation risk on the short leg.
DEFAULT_MARGIN_BUFFER: Final[float] = 0.02


def funding_apr(funding_rate: float, interval_hours: float) -> float:
    """Annualise one funding rate. The interval comes from the data, never assumed to be 8h (§5.1)."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    return funding_rate * (HOURS_PER_YEAR / interval_hours)


def entry_threshold_apr(round_trip_cost_bps: float, holding_days: float, margin_buffer: float) -> float:
    """§8.3: `2 * round_trip_cost_apr_equivalent + margin_buffer`.

    A cash-and-carry round trip pays the round-trip cost on *both* legs, so the cost to recover is
    twice a single round-trip. Converting that to an APR needs an assumed holding period: paying
    30 bps to earn funding for a week is a very different proposition from paying it to earn
    funding for a day, and the shorter the hold the higher the funding rate has to be.
    """
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    round_trip_cost = 2.0 * round_trip_cost_bps * 1e-4
    cost_apr_equivalent = round_trip_cost * (365.0 / holding_days)
    return 2.0 * cost_apr_equivalent + margin_buffer


class CarrySignal(Signal):
    """Funding cash-and-carry, per §8.3. Tier 1, and structurally non-directional."""

    name = "carry"
    tier = 1
    required_features: ClassVar[list[FeatureSpec]] = [FeatureSpec("funding_apr", 1)]
    # Two axes: how much funding is demanded before deploying, and how long a hold the entry
    # threshold is priced for. `margin_buffer` is a family constant.
    param_space: ClassVar[dict[str, ParamRange]] = {
        "min_holding_days": ParamRange("min_holding_days", (1.0, 3.0, 7.0)),
        "exit_fraction": ParamRange("exit_fraction", (0.0, 0.25, 0.5)),
    }

    def generate(self, features: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        """Emit sleeve deployment in [0, 1] from the funding APR carried on each bar.

        `features` must carry `open_time` and `funding_apr` — the annualised rate in force for that
        bar, forward-filled from the last settlement, which is what a trader would actually see.
        """
        if "funding_apr" not in features.columns:
            raise ValueError("CarrySignal needs a funding_apr column; join funding onto the bars")

        holding_days = float(params.get("min_holding_days", 3.0))
        exit_fraction = float(params.get("exit_fraction", 0.25))
        round_trip_bps = float(params.get("round_trip_cost_bps", 15.0))
        margin_buffer = float(params.get("margin_buffer", DEFAULT_MARGIN_BUFFER))

        entry = entry_threshold_apr(round_trip_bps, holding_days, margin_buffer)
        exit_level = entry * exit_fraction

        # Hysteresis as a running state: deployed once funding clears `entry`, stays deployed until
        # it falls below `exit_level`. Expressed as a cumulative fold so it stays causal and
        # vectorised — row i depends only on rows <= i.
        apr = features["funding_apr"].fill_null(0.0).to_numpy()
        deployed = _hysteresis(apr, entry, exit_level)

        out = features.select(pl.col("open_time").alias("timestamp")).with_columns(
            pl.Series("target_position", deployed, dtype=pl.Float64),
            pl.Series("confidence", (apr / max(entry, 1e-12)).clip(0.0, 1.0), dtype=pl.Float64),
        )
        return self.validate_output(out)


def _hysteresis(values: np.ndarray, entry: float, exit_level: float) -> np.ndarray:
    """1.0 once `values` exceeds `entry`, back to 0.0 once it falls below `exit_level`.

    An explicit scan rather than a vectorised comparison, because the state is genuinely
    path-dependent: whether bar i is deployed depends on whether bar i-1 was. It is still causal —
    the scan only ever looks backwards.
    """
    out = np.zeros(len(values), dtype=float)
    on = False
    for i, value in enumerate(values):
        on = value > exit_level if on else value > entry
        out[i] = 1.0 if on else 0.0
    return out
