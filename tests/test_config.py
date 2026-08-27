from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cryptolab.config import BaseConfig, load_cost_settings, load_strategy

CONFIG = Path("config")


def test_base_config_loads():
    base = BaseConfig.load(CONFIG / "base.yaml")
    assert base.universe == ["BTCUSDT", "ETHUSDT"]
    assert base.splits.train_start < base.splits.validation_start < base.splits.test_start


def test_split_protocol_covers_the_declared_train_window():
    """The train window starts 2019-01-01 — 'three years of data' does not fill it."""
    base = BaseConfig.load(CONFIG / "base.yaml")
    span_days = (base.splits.train_end - base.splits.train_start) / 86_400_000
    assert span_days > 1400


def test_sealed_period_is_open_ended():
    assert BaseConfig.load(CONFIG / "base.yaml").splits.test_end is None


def test_costs_yaml_matches_the_fixed_model():
    assert load_cost_settings(CONFIG / "costs.yaml")["regimes"]["conservative"]["taker_fee_bps"] == 5.5


def test_a_cheapened_costs_yaml_is_refused(tmp_path):
    """The cost model is not configurable downward. Drift from the §7 table raises."""
    raw = yaml.safe_load((CONFIG / "costs.yaml").read_text())
    raw["regimes"]["conservative"]["taker_fee_bps"] = 1.0
    path = tmp_path / "costs.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not a tunable parameter"):
        load_cost_settings(path)


def test_unknown_regime_in_yaml_is_refused(tmp_path):
    raw = yaml.safe_load((CONFIG / "costs.yaml").read_text())
    raw["regimes"]["free_lunch"] = raw["regimes"]["optimistic"]
    path = tmp_path / "costs.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown regime"):
        load_cost_settings(path)


@pytest.mark.parametrize("name", ["tsmom", "regime", "carry"])
def test_every_tier_one_strategy_has_a_config(name):
    config = load_strategy(name, CONFIG)
    assert config["name"] == name
    assert config["tier"] == 1


def test_tsmom_config_declares_the_spec_parameter_space():
    params = load_strategy("tsmom", CONFIG)["params"]
    assert params["lookback_bars"] == [24, 48, 96, 168]
    assert params["vol_halflife"] == [36, 72, 144]


def test_strategies_default_to_conservative_costs():
    for name in ("tsmom", "regime", "carry"):
        assert load_strategy(name, CONFIG)["costs"] == "conservative"


def test_non_mapping_yaml_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_cost_settings(path)
