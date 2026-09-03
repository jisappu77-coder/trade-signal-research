"""Canonical polars schemas — the single source of truth for every table (SPEC.md §5.3).

All timestamps are int64 UTC milliseconds, bar-open convention (SPEC.md §5.2).
Every persisted table additionally carries `ingested_at` and `source_uri`.
"""

from __future__ import annotations

from typing import Final

import polars as pl

# Provenance columns appended to every persisted table.
PROVENANCE: Final[dict[str, pl.DataType]] = {
    "ingested_at": pl.Int64(),
    "source_uri": pl.Utf8(),
}

OHLCV: Final[dict[str, pl.DataType]] = {
    "open_time": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
    "taker_buy_base": pl.Float64(),
    "taker_buy_quote": pl.Float64(),
    "close_time": pl.Int64(),
}

FUNDING: Final[dict[str, pl.DataType]] = {
    "funding_time": pl.Int64(),
    "symbol": pl.Utf8(),
    "funding_rate": pl.Float64(),
    "mark_price": pl.Float64(),
}

OPEN_INTEREST: Final[dict[str, pl.DataType]] = {
    "timestamp": pl.Int64(),
    "symbol": pl.Utf8(),
    "oi_base": pl.Float64(),
    "oi_quote": pl.Float64(),
}

# Mark-price klines. Basis is computed from mark price, never last-traded price (§5.1).
MARK_PRICE: Final[dict[str, pl.DataType]] = {
    "open_time": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "close_time": pl.Int64(),
}

# Contract specification changes (§5.1). A backtest spanning an unhandled change is refused.
CONTRACT_SPEC_CHANGES: Final[dict[str, pl.DataType]] = {
    "symbol": pl.Utf8(),
    "effective_time": pl.Int64(),
    "field": pl.Utf8(),
    "old_value": pl.Utf8(),
    "new_value": pl.Utf8(),
    "handled": pl.Boolean(),
}

SCHEMAS: Final[dict[str, dict[str, pl.DataType]]] = {
    "ohlcv": OHLCV,
    "funding": FUNDING,
    "open_interest": OPEN_INTEREST,
    "mark_price": MARK_PRICE,
    "contract_spec_changes": CONTRACT_SPEC_CHANGES,
}

# The timestamp column that orders each dataset.
TIME_COLUMN: Final[dict[str, str]] = {
    "ohlcv": "open_time",
    "funding": "funding_time",
    "open_interest": "timestamp",
    "mark_price": "open_time",
    "contract_spec_changes": "effective_time",
}

BAR_INTERVAL_MS: Final[dict[str, int]] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class SchemaError(ValueError):
    """Raised when a frame does not conform to its canonical schema."""


def schema_for(dataset: str, *, with_provenance: bool = False) -> dict[str, pl.DataType]:
    """Return the canonical schema for `dataset`.

    Raises SchemaError for an unknown dataset rather than returning a silent empty dict.
    """
    if dataset not in SCHEMAS:
        raise SchemaError(f"unknown dataset {dataset!r}; known: {sorted(SCHEMAS)}")
    schema = dict(SCHEMAS[dataset])
    if with_provenance:
        schema.update(PROVENANCE)
    return schema


def validate(df: pl.DataFrame, dataset: str, *, with_provenance: bool = False) -> pl.DataFrame:
    """Assert `df` conforms to the canonical schema, returning it column-ordered.

    Checks presence, absence of extras, and dtype. Widening or reordering silently is exactly
    the sort of drift that makes two backtests incomparable, so this is strict.
    """
    expected = schema_for(dataset, with_provenance=with_provenance)
    actual = dict(df.schema)

    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if missing or extra:
        raise SchemaError(f"{dataset}: missing columns {missing}, unexpected columns {extra}")

    wrong = {c: (str(actual[c]), str(dt)) for c, dt in expected.items() if actual[c] != dt}
    if wrong:
        raise SchemaError(f"{dataset}: dtype mismatch {wrong} (got, expected)")

    return df.select(list(expected))


def empty(dataset: str, *, with_provenance: bool = False) -> pl.DataFrame:
    """An empty, correctly-typed frame for `dataset`."""
    return pl.DataFrame(schema=schema_for(dataset, with_provenance=with_provenance))
