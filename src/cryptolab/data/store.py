"""Partitioned Parquet lake with sealed-test-period enforcement (SPEC.md §5.2, §10.1).

Layout: `data/{dataset}/exchange={ex}/symbol={sym}/year={y}/month={m}/part.parquet`

Reads are range-checked against the split protocol. Any read whose window overlaps the sealed
test period is refused unless the caller presents a valid, unspent token (§10.1). There is no
flag, environment variable or config key that disables this.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cryptolab.data import schemas
from cryptolab.validation.sealed import SealedPeriodError, TestPeriodToken


def to_ms(value: str | dt.date | dt.datetime | int) -> int:
    """Convert a date-ish value to int64 UTC milliseconds (bar-open convention)."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    if isinstance(value, dt.datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    else:
        stamp = dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1000)


def from_ms(ms: int) -> dt.datetime:
    """Inverse of `to_ms`, as a timezone-aware UTC datetime."""
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.UTC)


@dataclass(frozen=True, slots=True)
class SplitProtocol:
    """The §10.1 train / validation / test windows, in UTC milliseconds."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int | None = None

    @staticmethod
    def from_config(splits: dict[str, dict[str, str | None]]) -> SplitProtocol:
        def bound(name: str, key: str) -> int | None:
            raw = splits[name][key]
            return None if raw is None else to_ms(raw)

        test_start = bound("test", "start")
        if test_start is None:
            raise ValueError("splits.test.start is required — the sealed period must have a start")
        train_start, train_end = bound("train", "start"), bound("train", "end")
        val_start, val_end = bound("validation", "start"), bound("validation", "end")
        if train_start is None or train_end is None or val_start is None or val_end is None:
            raise ValueError("train and validation windows must both be closed intervals")
        return SplitProtocol(train_start, train_end, val_start, val_end, test_start, bound("test", "end"))

    def touches_sealed(self, start_ms: int, end_ms: int) -> bool:
        """True if [start_ms, end_ms] overlaps the sealed test window at all."""
        if end_ms < self.test_start:
            return False
        return self.test_end is None or start_ms <= self.test_end


class ParquetStore:
    """Read/write access to the partitioned lake.

    `token_verifier` is injected by `validation.registry` so that the store can check a presented
    token without importing the registry (and so tests can exercise refusal without a registry).
    """

    def __init__(
        self,
        root: Path | str,
        splits: SplitProtocol,
        *,
        token_verifier: Callable[[TestPeriodToken], bool] | None = None,
    ) -> None:
        self.root = Path(root)
        self.splits = splits
        self._verify = token_verifier

    # ---- paths -------------------------------------------------------------------

    def partition_dir(self, dataset: str, exchange: str, symbol: str, year: int, month: int) -> Path:
        return (
            self.root
            / dataset
            / f"exchange={exchange}"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month:02d}"
        )

    def dataset_dir(self, dataset: str, exchange: str, symbol: str) -> Path:
        return self.root / dataset / f"exchange={exchange}" / f"symbol={symbol}"

    # ---- write -------------------------------------------------------------------

    def write(
        self,
        df: pl.DataFrame,
        dataset: str,
        *,
        exchange: str,
        symbol: str,
        source_uri: str,
        ingested_at: int | None = None,
    ) -> list[Path]:
        """Validate, stamp provenance, and write `df` partitioned by year/month.

        Rewrites whole month partitions — ingestion is idempotent per month by construction.
        """
        df = schemas.validate(df, dataset)
        stamped = df.with_columns(
            pl.lit(ingested_at if ingested_at is not None else _now_ms(), dtype=pl.Int64).alias(
                "ingested_at"
            ),
            pl.lit(source_uri, dtype=pl.Utf8).alias("source_uri"),
        )
        time_col = schemas.TIME_COLUMN[dataset]
        keyed = stamped.with_columns(
            pl.col(time_col).cast(pl.Datetime("ms")).dt.year().alias("_year"),
            pl.col(time_col).cast(pl.Datetime("ms")).dt.month().alias("_month"),
        )
        written: list[Path] = []
        for (year, month), part in keyed.group_by(["_year", "_month"], maintain_order=True):
            target = self.partition_dir(dataset, exchange, symbol, int(year), int(month))
            target.mkdir(parents=True, exist_ok=True)
            path = target / "part.parquet"
            part.drop("_year", "_month").sort(time_col).write_parquet(path)
            written.append(path)
        return written

    # ---- read --------------------------------------------------------------------

    def read(
        self,
        dataset: str,
        *,
        exchange: str,
        symbol: str,
        start: str | dt.date | int,
        end: str | dt.date | int,
        token: TestPeriodToken | None = None,
    ) -> pl.DataFrame:
        """Read `[start, end]` for one symbol, enforcing the sealed-period rule.

        Raises SealedPeriodError if the window touches the test period without a valid token.
        """
        start_ms, end_ms = to_ms(start), to_ms(end)
        if end_ms < start_ms:
            raise ValueError(f"end ({end_ms}) precedes start ({start_ms})")

        if self.splits.touches_sealed(start_ms, end_ms):
            self._authorise_sealed(token, dataset, symbol)

        base = self.dataset_dir(dataset, exchange, symbol)
        if not base.exists():
            return schemas.empty(dataset, with_provenance=True)
        parts = sorted(base.glob("year=*/month=*/part.parquet"))
        if not parts:
            return schemas.empty(dataset, with_provenance=True)

        time_col = schemas.TIME_COLUMN[dataset]
        frame = (
            pl.scan_parquet(parts)
            .filter(pl.col(time_col).is_between(start_ms, end_ms, closed="both"))
            .sort(time_col)
            .collect()
        )
        return schemas.validate(frame, dataset, with_provenance=True)

    def _authorise_sealed(self, token: TestPeriodToken | None, dataset: str, symbol: str) -> None:
        if token is None:
            raise SealedPeriodError(
                f"read of {dataset}/{symbol} touches the sealed test period "
                f"(from {from_ms(self.splits.test_start).date()}). A one-time token issued by "
                "validation.registry.TrialRegistry.issue_test_token is required (SPEC.md §10.1)."
            )
        if self._verify is None:
            raise SealedPeriodError(
                "a token was presented but this store has no verifier bound; construct the store "
                "via TrialRegistry.bind_store so the token can be checked against the registry"
            )
        if not self._verify(token):
            raise SealedPeriodError(
                f"token for strategy family {token.strategy_family!r} is invalid, already spent, "
                "or was issued by a different registry"
            )

    def read_universe(
        self,
        dataset: str,
        *,
        exchange: str,
        symbols: Iterable[str],
        start: str | dt.date | int,
        end: str | dt.date | int,
        token: TestPeriodToken | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Read the same window for several symbols. One token authorises the whole universe read."""
        return {
            symbol: self.read(dataset, exchange=exchange, symbol=symbol, start=start, end=end, token=token)
            for symbol in symbols
        }


def _now_ms() -> int:
    return int(dt.datetime.now(tz=dt.UTC).timestamp() * 1000)
