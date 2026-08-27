from __future__ import annotations

import polars as pl
import pytest

from cryptolab.data import schemas
from cryptolab.data.schemas import SchemaError


def test_every_dataset_has_a_time_column():
    assert set(schemas.SCHEMAS) == set(schemas.TIME_COLUMN)


def test_validate_rejects_missing_column(bars):
    with pytest.raises(SchemaError, match="missing columns"):
        schemas.validate(bars.drop("trades"), "ohlcv")


def test_validate_rejects_extra_column(bars):
    with pytest.raises(SchemaError, match="unexpected columns"):
        schemas.validate(bars.with_columns(pl.lit(1).alias("surprise")), "ohlcv")


def test_validate_rejects_wrong_dtype(bars):
    with pytest.raises(SchemaError, match="dtype mismatch"):
        schemas.validate(bars.with_columns(pl.col("trades").cast(pl.Float64)), "ohlcv")


def test_validate_reorders_to_canonical_order(bars):
    shuffled = bars.select(sorted(bars.columns))
    assert schemas.validate(shuffled, "ohlcv").columns == list(schemas.OHLCV)


def test_unknown_dataset_raises():
    with pytest.raises(SchemaError, match="unknown dataset"):
        schemas.schema_for("nope")


def test_empty_frame_matches_schema():
    assert schemas.empty("funding").schema == schemas.FUNDING
