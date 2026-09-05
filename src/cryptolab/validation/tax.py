"""Indian VDA tax treatment (SPEC.md §17).

§17 requires the backtester to report **pre-tax and post-tax net returns separately**, because a
high-turnover strategy taxed at 30% on gross gains with no loss offset has materially different
economics from its pre-tax backtest.

Two effects, and conflating them is the easy mistake:

* **§115BBH — 30% flat, no set-off, no carry-forward.** This is a *permanent* asymmetry. Losses buy
  nothing: they cannot offset other VDA gains, other income, or future years. A grid where some
  configurations win and others lose is taxed on the winners alone, so the after-tax total is worse
  than netting suggests. This is the effect that changes whether a strategy is worth running.

* **§194S — 1% TDS per disposal.** This is *withholding*, creditable against final liability — not
  a permanent cost. At high turnover the amount withheld across a year can exceed the account's
  capital, recoverable only by refund, which is a real financing and cash-flow constraint. It is
  **not** a 1% loss per trade, and modelling it as one overstates the drag by an order of magnitude.

The TDS provision is §194S, not §115BBH — a common mis-citation, flagged in §17 itself.

This models economics. It is not tax advice; §17 requires confirmation with a qualified Indian tax
professional before any deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

VDA_RATE: Final[float] = 0.30  # §115BBH statutory rate, as §17 states it
TDS_RATE: Final[float] = 0.01  # §194S, on the transfer value of each disposal

# Health & Education Cess, levied on the tax itself rather than on income. §17 names only the 30%
# statutory rate, so this is a deliberate refinement of the spec rather than a restatement of it:
# the cess is a real liability, and omitting it flatters every post-tax figure this project prints.
# It errs conservative, which is the safe direction for a number used to decide whether to trade.
CESS_RATE: Final[float] = 0.04
EFFECTIVE_VDA_RATE: Final[float] = VDA_RATE * (1.0 + CESS_RATE)

# ---- the benchmark side ------------------------------------------------------------------
#
# **The comparison that was wrong, and why it matters.** A post-tax VDA return must be compared
# against a post-tax alternative. Indian fixed-deposit interest is taxed as *income from other
# sources* at the holder's slab rate — it is not a risk-free 7% for anyone who pays income tax.
# Comparing a 30%-taxed strategy return against an untaxed deposit rate overstates the hurdle by
# up to a third and makes a marginal strategy look hopeless.
DEFAULT_FD_RATE_APR: Final[float] = 0.07
# The slab to assume when none is given. The top bracket is the conservative choice for a hurdle:
# it is the *lowest* post-tax deposit return, so it is the easiest bar for a strategy to clear, and
# a strategy that fails even this one fails for everybody.
DEFAULT_SLAB_RATE: Final[float] = 0.30


def fixed_deposit_hurdle_apr(
    fd_rate_apr: float = DEFAULT_FD_RATE_APR,
    slab_rate: float = DEFAULT_SLAB_RATE,
) -> float:
    """Post-tax return of the risk-free alternative, at a given deposit rate and income slab.

    At a 30% slab plus cess a 7% deposit returns 4.82% post-tax, not 7%. Only a holder below the
    rebate threshold (slab 0) actually keeps the headline rate.
    """
    if not 0.0 <= slab_rate < 1.0:
        raise ValueError(f"slab_rate must be in [0, 1), got {slab_rate}")
    return fd_rate_apr * (1.0 - slab_rate * (1.0 + CESS_RATE))


@dataclass(frozen=True, slots=True)
class TaxOutcome:
    """Pre- and post-tax economics for one run or one portfolio of runs."""

    pre_tax_pnl: float
    taxable_gains: float
    unusable_losses: float
    tax_due: float
    tds_withheld: float
    initial_equity: float
    years: float

    @property
    def post_tax_pnl(self) -> float:
        return self.pre_tax_pnl - self.tax_due

    @property
    def pre_tax_return(self) -> float:
        return self.pre_tax_pnl / self.initial_equity if self.initial_equity else 0.0

    @property
    def post_tax_return(self) -> float:
        return self.post_tax_pnl / self.initial_equity if self.initial_equity else 0.0

    @property
    def effective_rate(self) -> float:
        """Tax as a share of pre-tax profit. Exceeds the headline rate whenever losses are stranded."""
        return self.tax_due / self.pre_tax_pnl if self.pre_tax_pnl > 0 else 0.0

    @property
    def tds_as_multiple_of_capital(self) -> float:
        """Annual withholding relative to the account. Above 1.0 the refund cycle is the business."""
        if self.initial_equity <= 0 or self.years <= 0:
            return 0.0
        return self.tds_withheld / self.years / self.initial_equity


def tax_single_run(
    *,
    pre_tax_pnl: float,
    traded_notional: float,
    initial_equity: float,
    years: float,
) -> TaxOutcome:
    """Tax one run. A loss produces no relief, which is the §115BBH asymmetry in one line."""
    gains = max(pre_tax_pnl, 0.0)
    losses = -min(pre_tax_pnl, 0.0)
    # TDS applies to the disposal side. `traded_notional` counts every one-way fill, so half of it
    # is sells.
    return TaxOutcome(
        pre_tax_pnl=pre_tax_pnl,
        taxable_gains=gains,
        unusable_losses=losses,
        tax_due=EFFECTIVE_VDA_RATE * gains,
        tds_withheld=TDS_RATE * traded_notional / 2.0,
        initial_equity=initial_equity,
        years=years,
    )


def tax_portfolio(
    pnls: Sequence[float],
    *,
    traded_notional: float,
    initial_equity: float,
    years: float,
) -> TaxOutcome:
    """Tax a set of runs held together, with **no set-off between them**.

    This is where §115BBH bites hardest and where netting quietly flatters a result: each winner is
    taxed in full while every loser is stranded, so the effective rate on total profit exceeds 30%
    whenever the set contains a loss.
    """
    total = float(sum(pnls))
    gains = float(sum(p for p in pnls if p > 0))
    losses = float(-sum(p for p in pnls if p < 0))
    return TaxOutcome(
        pre_tax_pnl=total,
        taxable_gains=gains,
        unusable_losses=losses,
        tax_due=EFFECTIVE_VDA_RATE * gains,
        tds_withheld=TDS_RATE * traded_notional / 2.0,
        initial_equity=initial_equity,
        years=years,
    )


def summary_line(outcome: TaxOutcome) -> str:
    """The §17 statement, computed rather than hand-written."""
    if outcome.pre_tax_pnl <= 0:
        return (
            f"pre-tax {outcome.pre_tax_return:+.1%}; no tax due on a loss, and under §115BBH that "
            "loss cannot be set off against other income or carried forward"
        )
    stranded = (
        f", with {outcome.unusable_losses:,.0f} of losses stranded (no set-off under §115BBH)"
        if outcome.unusable_losses > 0
        else ""
    )
    return (
        f"pre-tax {outcome.pre_tax_return:+.1%} → post-tax {outcome.post_tax_return:+.1%} "
        f"at an effective {outcome.effective_rate:.0%} rate{stranded}; TDS withheld "
        f"{outcome.tds_as_multiple_of_capital:.1f}x capital per year (creditable, but financed)"
    )
