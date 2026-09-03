from __future__ import annotations

import pytest

from cryptolab.data.store import ParquetStore, SplitProtocol, to_ms
from cryptolab.validation.synthetic import synthetic_bars

SPLITS_YAML = {
    "train": {"start": "2019-01-01", "end": "2022-12-31"},
    "validation": {"start": "2023-01-01", "end": "2024-06-30"},
    "test": {"start": "2024-07-01", "end": None},
}


@pytest.fixture
def splits() -> SplitProtocol:
    return SplitProtocol.from_config(SPLITS_YAML)


@pytest.fixture
def store(tmp_path, splits) -> ParquetStore:
    return ParquetStore(tmp_path / "data", splits)


@pytest.fixture
def bars():
    return synthetic_bars(2000, seed=42)


@pytest.fixture
def train_ms():
    return to_ms("2019-01-01"), to_ms("2022-12-31")
