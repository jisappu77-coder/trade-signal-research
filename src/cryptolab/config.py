"""Configuration loading (SPEC.md §16). YAML in, typed objects out, no module-level I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cryptolab.data.store import SplitProtocol

DEFAULT_CONFIG_DIR = Path("config")


@dataclass(frozen=True, slots=True)
class BaseConfig:
    """`config/base.yaml` — universe, paths, and the split protocol."""

    universe: list[str]
    exchange: str
    bars: list[str]
    data_root: Path
    registry_path: Path
    report_root: Path
    splits: SplitProtocol

    @staticmethod
    def load(path: Path | str = DEFAULT_CONFIG_DIR / "base.yaml") -> BaseConfig:
        raw = _read_yaml(path)
        return BaseConfig(
            universe=list(raw["universe"]),
            exchange=str(raw["exchange"]),
            bars=list(raw["bars"]),
            data_root=Path(raw["data_root"]),
            registry_path=Path(raw["registry_path"]),
            report_root=Path(raw["report_root"]),
            splits=SplitProtocol.from_config(raw["splits"]),
        )


def load_strategy(name: str, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    """Load one strategy YAML from `config/strategies/`."""
    return _read_yaml(Path(config_dir) / "strategies" / f"{name}.yaml")


def load_cost_settings(path: Path | str = DEFAULT_CONFIG_DIR / "costs.yaml") -> dict[str, Any]:
    """Load `costs.yaml` and verify it still matches the fixed §7 regimes.

    The YAML exists so the numbers are visible in one place, not so they can be changed. Any drift
    from the constants in `backtest.costs` is a protocol violation and raises here.
    """
    # Imported here so `config` stays importable from `backtest.costs` without a cycle.
    from cryptolab.backtest.costs import REGIMES  # noqa: PLC0415

    raw = _read_yaml(path)
    for name, values in raw["regimes"].items():
        regime = REGIMES.get(name)
        if regime is None:
            raise ValueError(f"costs.yaml declares unknown regime {name!r}")
        drift = {
            field: (values[field], getattr(regime, field))
            for field in ("taker_fee_bps", "maker_fee_bps", "slippage_bps", "funding_multiplier")
            if abs(float(values[field]) - float(getattr(regime, field))) > 1e-12
        }
        if drift:
            raise ValueError(
                f"costs.yaml regime {name!r} has drifted from the fixed §7 model: {drift} "
                "(yaml, code). The cost model is not a tunable parameter."
            )
    return raw


def _read_yaml(path: Path | str) -> dict[str, Any]:
    text = Path(path).read_text()
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(parsed).__name__}")
    return parsed
