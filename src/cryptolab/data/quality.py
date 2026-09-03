"""The §6 data-quality gate.

Runs before any feature computation and **hard-fails** the pipeline. A backtest over data that
failed this gate is not a result — the engine records the hash of every quality report it consumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import polars as pl

from cryptolab.data.schemas import BAR_INTERVAL_MS

Severity = Literal["fail", "warn"]

# Exchange funding cap; a rate beyond this is bad data, not a market event (§6).
FUNDING_CAP = 0.0075
MAX_ZERO_VOLUME_RUN = 10
PRICE_JUMP_THRESHOLD = 0.20
VOLUME_SPIKE_MULTIPLE = 3.0


class DataQualityError(RuntimeError):
    """Raised when the quality gate fails. The pipeline stops here, by design."""


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """One violation. `location` is a timestamp or range; `detail` carries counts/durations."""

    check: str
    severity: Severity
    count: int
    location: str
    detail: str


@dataclass(slots=True)
class DataQualityReport:
    """Per symbol-month quality verdict, persisted and hashed (§6)."""

    dataset: str
    symbol: str
    interval: str | None
    rows: int
    start_time: int | None
    end_time: int | None
    findings: list[QualityFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "fail" for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "symbol": self.symbol,
            "interval": self.interval,
            "rows": self.rows,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "passed": self.passed,
            "findings": [asdict(f) for f in self.findings],
        }

    def content_hash(self) -> str:
        """Stable hash of the report, recorded by every backtest that consumed this data."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        lines = [
            f"  [{f.check}] {f.count}x at {f.location}: {f.detail}"
            for f in self.findings
            if f.severity == "fail"
        ]
        raise DataQualityError(
            f"data quality gate FAILED for {self.dataset}/{self.symbol} "
            f"({self.interval or 'n/a'}):\n" + "\n".join(lines)
        )


def check_ohlcv(df: pl.DataFrame, symbol: str, interval: str) -> DataQualityReport:
    """Run every §6 OHLCV check. Returns the report; the caller decides when to raise."""
    if interval not in BAR_INTERVAL_MS:
        raise ValueError(f"unknown interval {interval!r}")
    step = BAR_INTERVAL_MS[interval]
    findings: list[QualityFinding] = []
    report = DataQualityReport(
        dataset="ohlcv",
        symbol=symbol,
        interval=interval,
        rows=df.height,
        start_time=int(df["open_time"].min()) if df.height else None,  # type: ignore[arg-type]
        end_time=int(df["open_time"].max()) if df.height else None,  # type: ignore[arg-type]
        findings=findings,
    )
    if df.height == 0:
        findings.append(QualityFinding("empty", "fail", 0, "-", "no rows"))
        return report

    times = df["open_time"]

    # Duplicate timestamps.
    dupes = df.group_by("open_time").len().filter(pl.col("len") > 1)
    if dupes.height:
        findings.append(
            QualityFinding(
                "duplicate_timestamps",
                "fail",
                dupes.height,
                str(int(dupes["open_time"].min())),  # type: ignore[arg-type]
                f"{dupes.height} timestamps occur more than once",
            )
        )

    # Non-monotonic timestamps.
    if not times.is_sorted():
        diffs = times.diff().drop_nulls()
        backwards = int((diffs < 0).sum())
        findings.append(
            QualityFinding("non_monotonic", "fail", backwards, "-", f"{backwards} backward timestamp steps")
        )

    # Missing bars — report count, location and duration.
    gaps = (
        df.sort("open_time")
        .select(
            pl.col("open_time").alias("t"),
            (pl.col("open_time").diff() - step).alias("excess"),
        )
        .filter(pl.col("excess") > 0)
    )
    if gaps.height:
        worst = gaps.sort("excess", descending=True).row(0, named=True)
        missing_bars = int((gaps["excess"] // step).sum())
        findings.append(
            QualityFinding(
                "missing_bars",
                "fail",
                gaps.height,
                f"largest before open_time={worst['t']}",
                f"{missing_bars} bars missing across {gaps.height} gaps; "
                f"largest gap {int(worst['excess']) // 60_000} minutes",
            )
        )

    # OHLC violations.
    bad_ohlc = df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.col("low"))
    )
    if bad_ohlc.height:
        findings.append(
            QualityFinding(
                "ohlc_violation",
                "fail",
                bad_ohlc.height,
                str(int(bad_ohlc["open_time"].min())),  # type: ignore[arg-type]
                "high < max(open, close), low > min(open, close), or high < low",
            )
        )

    # Zero-volume runs > 10 bars: an outage, not a quiet market.
    run_len = _max_zero_run(df.sort("open_time")["volume"])
    if run_len > MAX_ZERO_VOLUME_RUN:
        findings.append(
            QualityFinding(
                "zero_volume_run",
                "fail",
                run_len,
                "-",
                f"{run_len} consecutive zero-volume bars (> {MAX_ZERO_VOLUME_RUN}) indicates an outage",
            )
        )

    # Price jumps > 20% in one bar without a corresponding volume spike.
    jumps = (
        df.sort("open_time")
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1)) - 1).abs().alias("ret"),
            (pl.col("volume") / pl.col("volume").rolling_mean(24, min_samples=2)).alias("vol_ratio"),
        )
        .filter(
            (pl.col("ret") > PRICE_JUMP_THRESHOLD)
            & (pl.col("vol_ratio").fill_null(0.0) < VOLUME_SPIKE_MULTIPLE)
        )
    )
    if jumps.height:
        findings.append(
            QualityFinding(
                "unsupported_price_jump",
                "fail",
                jumps.height,
                str(int(jumps["open_time"].min())),  # type: ignore[arg-type]
                f">{PRICE_JUMP_THRESHOLD:.0%} move with no volume confirmation",
            )
        )

    return report


def check_funding(df: pl.DataFrame, symbol: str) -> DataQualityReport:
    """Funding-rate checks: duplicates, monotonicity, and the exchange cap breach (§6)."""
    findings: list[QualityFinding] = []
    report = DataQualityReport(
        dataset="funding",
        symbol=symbol,
        interval=None,
        rows=df.height,
        start_time=int(df["funding_time"].min()) if df.height else None,  # type: ignore[arg-type]
        end_time=int(df["funding_time"].max()) if df.height else None,  # type: ignore[arg-type]
        findings=findings,
    )
    if df.height == 0:
        findings.append(QualityFinding("empty", "fail", 0, "-", "no rows"))
        return report

    dupes = df.group_by("funding_time").len().filter(pl.col("len") > 1)
    if dupes.height:
        findings.append(
            QualityFinding("duplicate_timestamps", "fail", dupes.height, "-", "repeated funding_time")
        )
    if not df["funding_time"].is_sorted():
        findings.append(QualityFinding("non_monotonic", "fail", 1, "-", "funding_time not sorted"))

    breaches = df.filter(pl.col("funding_rate").abs() > FUNDING_CAP)
    if breaches.height:
        findings.append(
            QualityFinding(
                "funding_cap_breach",
                "fail",
                breaches.height,
                str(int(breaches["funding_time"].min())),  # type: ignore[arg-type]
                f"|funding_rate| exceeds the {FUNDING_CAP:.2%} exchange cap — bad data, not a market event",
            )
        )
    return report


def _max_zero_run(series: pl.Series) -> int:
    """Longest run of exactly-zero values."""
    is_zero = series == 0
    if not bool(is_zero.any()):
        return 0
    # Group consecutive equal values, then take the longest True group.
    groups = (is_zero != is_zero.shift(1)).fill_null(True).cum_sum()
    runs = (
        pl.DataFrame({"zero": is_zero, "grp": groups})
        .group_by("grp")
        .agg(pl.col("zero").first().alias("zero"), pl.len().alias("n"))
        .filter(pl.col("zero"))
    )
    return int(runs["n"].max()) if runs.height else 0  # type: ignore[arg-type]
