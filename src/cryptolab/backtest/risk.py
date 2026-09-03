"""The risk engine (SPEC.md §13).

Independent of signal logic and able to veto any target position. The kill switch is a first-class
object with an audit log, exercised in backtest so the interface is proven before any live wiring —
even though v1 places no orders.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class KillReason(StrEnum):
    """Why the kill switch fired. Recorded verbatim in the audit log."""

    DAILY_LOSS = "daily_loss_limit"
    DRAWDOWN = "drawdown_limit"
    CONSECUTIVE_LOSSES = "max_consecutive_losses"
    STALE_DATA = "data_staleness"
    LEVERAGE = "max_gross_leverage"
    VOLATILITY = "volatility_circuit_breaker"
    INSOLVENT = "insolvent"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """§13 limits. Defaults are deliberately tight; loosening them is a recorded decision."""

    max_gross_leverage: float = 2.0
    max_position_per_symbol: float = 1.0
    daily_loss_limit: float = 0.05
    drawdown_limit: float = 0.35
    max_consecutive_losses: int = 8
    staleness_intervals: int = 2
    volatility_circuit_multiple: float = 4.0
    # A fully-invested book sits exactly at `max_gross_leverage`, and two things push it over
    # without anything having gone wrong: marking to market, and the §9.3 no-trade band, which
    # deliberately lets exposure drift before rebalancing. The tolerance must therefore exceed the
    # band in use (default 0.10) or the band alone guarantees a kill on every fully-invested book.
    leverage_tolerance: float = 0.15


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One immutable line in the kill-switch audit log."""

    timestamp: int
    reason: KillReason
    detail: str
    action: Literal["flatten", "veto", "reset"]


@dataclass(slots=True)
class KillSwitch:
    """First-class kill switch with an append-only audit log."""

    tripped: bool = False
    reason: KillReason | None = None
    audit_log: list[AuditEntry] = field(default_factory=list)

    def trip(self, timestamp: int, reason: KillReason, detail: str) -> None:
        """Fire the switch. Idempotent — the first reason is the one that sticks."""
        if not self.tripped:
            self.tripped = True
            self.reason = reason
        self.audit_log.append(AuditEntry(timestamp, reason, detail, "flatten"))

    def reset(self, timestamp: int, detail: str = "manual reset") -> None:
        """Clear the switch, recording the reset in the audit log."""
        self.tripped = False
        self.reason = None
        self.audit_log.append(AuditEntry(timestamp, KillReason.MANUAL, detail, "reset"))


@dataclass(slots=True)
class RiskEngine:
    """Evaluates limits each bar and may force a flatten (step 3 of the §9.2 event loop)."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    kill_switch: KillSwitch = field(default_factory=KillSwitch)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    current_day: int = -1
    consecutive_losses: int = 0

    def observe(self, timestamp: int, equity: float) -> None:
        """Update running state. Call once per bar, before `check`."""
        day = timestamp // 86_400_000
        if day != self.current_day:
            self.current_day = day
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def record_trade_result(self, pnl: float) -> None:
        """Track the consecutive-loss counter."""
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0

    def check(
        self,
        timestamp: int,
        equity: float,
        *,
        bars_since_last_data: int = 0,
        realised_vol: float | None = None,
        baseline_vol: float | None = None,
        gross_notional: float | None = None,
    ) -> bool:
        """Return True if the engine demands a flatten. Trips the kill switch with a reason.

        `gross_notional` is the *realised* exposure. Checking it matters: clamping the target alone
        leaves the limit unenforced when the book cannot delever, and realised leverage then runs
        away from a target that looks obedient.

        Limits are evaluated in severity order and the first breach wins, so the recorded reason is
        the most serious one rather than whichever happened to be checked first.
        """
        for reason, breached, detail in self._breaches(
            equity,
            bars_since_last_data=bars_since_last_data,
            realised_vol=realised_vol,
            baseline_vol=baseline_vol,
            gross_notional=gross_notional,
        ):
            if breached:
                self.kill_switch.trip(timestamp, reason, detail)
                return True
        return False

    def _breaches(
        self,
        equity: float,
        *,
        bars_since_last_data: int,
        realised_vol: float | None,
        baseline_vol: float | None,
        gross_notional: float | None,
    ) -> list[tuple[KillReason, bool, str]]:
        """Every limit as a (reason, breached, detail) triple, most severe first."""
        leverage = abs(gross_notional) / equity if gross_notional is not None and equity > 0 else 0.0
        day_loss = (
            (equity - self.day_start_equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0
        )
        drawdown = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        vol_ratio = (
            realised_vol / baseline_vol
            if realised_vol is not None and baseline_vol is not None and baseline_vol > 0
            else 0.0
        )
        limits = self.limits
        return [
            (KillReason.INSOLVENT, equity <= 0, f"equity {equity:,.0f} is not positive"),
            (
                KillReason.LEVERAGE,
                leverage > limits.max_gross_leverage * (1.0 + limits.leverage_tolerance),
                f"realised gross leverage {leverage:.1f}x exceeds {limits.max_gross_leverage:.1f}x",
            ),
            (
                KillReason.DAILY_LOSS,
                day_loss <= -limits.daily_loss_limit,
                f"day P&L {day_loss:.2%}",
            ),
            (KillReason.DRAWDOWN, drawdown <= -limits.drawdown_limit, f"drawdown {drawdown:.2%}"),
            (
                KillReason.CONSECUTIVE_LOSSES,
                self.consecutive_losses >= limits.max_consecutive_losses,
                f"{self.consecutive_losses} consecutive losing trades",
            ),
            (
                KillReason.STALE_DATA,
                bars_since_last_data > limits.staleness_intervals,
                f"no fresh bar for {bars_since_last_data} intervals",
            ),
            (
                KillReason.VOLATILITY,
                vol_ratio > limits.volatility_circuit_multiple,
                f"realised vol {vol_ratio:.1f}x baseline",
            ),
        ]

    def clamp(self, target_position: float) -> float:
        """Apply per-symbol and gross-leverage caps to a desired exposure."""
        if self.kill_switch.tripped:
            return 0.0
        cap = min(self.limits.max_position_per_symbol, self.limits.max_gross_leverage)
        return max(-cap, min(cap, target_position))


def audit_log_as_text(entries: list[AuditEntry]) -> str:
    """Render the audit log for the report. Empty logs say so explicitly rather than printing nothing."""
    if not entries:
        return "kill switch: never fired"
    return "\n".join(
        f"{dt.datetime.fromtimestamp(e.timestamp / 1000, tz=dt.UTC):%Y-%m-%d %H:%M} "
        f"{e.reason.value} ({e.action}): {e.detail}"
        for e in entries
    )
