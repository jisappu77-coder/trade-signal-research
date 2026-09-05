from __future__ import annotations

import polars as pl
import pytest

from cryptolab.signals.base import FeatureSpec, ParamRange, Signal
from cryptolab.validation.synthetic import MomentumProbe, RandomSignal, ZeroSignal


class BadSignal(Signal):
    name = "bad"
    tier = 3
    required_features = [FeatureSpec("close", 1)]
    param_space: dict[str, ParamRange] = {}

    def generate(self, features, params):
        return self.validate_output(
            features.select(
                pl.col("open_time").alias("timestamp"),
                pl.lit(3.0).alias("target_position"),
                pl.lit(1.0).alias("confidence"),
            )
        )


def test_grid_is_the_full_declared_product():
    signal = MomentumProbe()
    assert len(signal.grid()) == 4


def test_tsmom_declared_space_is_twenty_four_combinations():
    """§8.1: L(4) x H(3) x bar(2) = 24 — the entire declared space."""

    class Space(Signal):
        name = "tsmom_space"
        tier = 1
        required_features = [FeatureSpec("close", 168)]
        param_space = {
            "L": ParamRange("L", (24, 48, 96, 168)),
            "H": ParamRange("H", (36, 72, 144)),
            "bar": ParamRange("bar", ("1h", "4h")),
        }

        def generate(self, features, params):
            raise NotImplementedError

    assert len(Space().grid()) == 24


def test_grid_order_is_stable():
    signal = MomentumProbe()
    assert signal.grid() == signal.grid()


def test_empty_param_space_yields_one_trial():
    assert ZeroSignal().grid() == [{}]


def test_max_lookback_is_the_declared_warmup():
    assert MomentumProbe().max_lookback_bars == 96


def test_out_of_range_target_is_refused(bars):
    with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
        BadSignal().generate(bars, {})


def test_output_contract_is_enforced(bars):
    signal = ZeroSignal()
    out = signal.generate(bars, {})
    assert out.columns == ["timestamp", "target_position", "confidence"]
    assert out.schema["timestamp"] == pl.Int64


def test_missing_column_is_refused():
    signal = ZeroSignal()
    with pytest.raises(ValueError, match="missing columns"):
        signal.validate_output(pl.DataFrame({"timestamp": [1, 2]}))


def test_unsorted_timestamps_are_refused():
    signal = ZeroSignal()
    frame = pl.DataFrame({"timestamp": [2, 1], "target_position": [0.0, 0.0], "confidence": [1.0, 1.0]})
    with pytest.raises(ValueError, match="unsorted timestamps"):
        signal.validate_output(frame)


def test_random_signal_is_reproducible(bars):
    signal = RandomSignal()
    first = signal.generate(bars, {"seed": 1})
    second = signal.generate(bars, {"seed": 1})
    assert first.equals(second)
    assert not first.equals(signal.generate(bars, {"seed": 2}))


def test_nan_target_is_rejected_with_a_useful_message(bars):
    """A vol-scaled signal reaches NaN via 0 * inf when sigma is zero; the message must say so."""
    signal = ZeroSignal()
    frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "target_position": [0.5, float("nan"), 0.5],
            "confidence": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="NaN target_position"):
        signal.validate_output(frame)
