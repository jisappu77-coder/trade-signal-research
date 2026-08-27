# cryptolab

An evidence-gated research instrument for intraday crypto perpetual futures.

Its output is a **verdict** on whether an edge exists — not a P&L, and not a trading bot. See
[`SPEC.md`](SPEC.md) for what it builds and [`CLAUDE.md`](CLAUDE.md) for how work on it is conducted.

## Status: Phases 1–3 complete

| Phase | Deliverable | State |
|-------|-------------|-------|
| 1 | Data layer: ingest, store, quality gate | done |
| 2 | Cost model + backtest engine | done |
| 3 | Validation harness | done |
| 4 | TSMOM | not started |
| 5 | REGIME overlay | not started |
| 6 | CARRY | not started |
| 7 | Reporting + comparison index | not started |

**No Tier-1 signal exists yet, and that is deliberate.** SPEC.md §15 requires the validation harness
to be green before the first signal is written. The CLI has no `run-strategy` command for the same
reason.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
cryptolab init            # config check + the constraints that bite
cryptolab harness-check   # exercise DSR / PBO / walk-forward on synthetic strategies
pytest tests/ -q
```

Ingest (bulk history comes from the public archive, never the REST API):

```bash
cryptolab ingest BTCUSDT --interval 1h --start 2019-01-01 --end 2024-06-30
cryptolab ingest-funding BTCUSDT --start 2019-01-01 --end 2024-06-30
cryptolab collect-oi      # run daily — see the warning below
```

## What is enforced in code, not by convention

- **The sealed test period.** `store.read` refuses any window touching 2024-07-01 onward without a
  one-time token from the trial registry. A second token for the same strategy family raises; a
  spent token raises; a token from another registry raises.
- **Trial counting.** The registry is append-only and hash-chained, and `N` for the deflated Sharpe
  is read from it. Trials are counted **per symbol**: 24 parameter combinations across two assets is
  N=48. Deleting or editing a row is detectable by `verify_chain`.
- **The cost model.** The four §7 regimes are frozen constants. `config/costs.yaml` is checked
  against them at load time and a cheapened value raises rather than loads.
- **Bar-close discipline.** A target stamped at bar `t` is joined onto bar `t+1` before the loop
  runs, so the engine physically cannot fill inside the signal bar.
- **Gate evaluation.** `evaluate_gates` refuses any regime but `conservative`, and the verdict line
  is computed, never written by hand.
- **DSR units.** The deflated-Sharpe functions reject inputs that look annualised; annualisation
  happens only at the reporting boundary.

## Validated against real data

[`docs/real-data-validation.md`](docs/real-data-validation.md) records a run over **54 months of
real Binance BTC and ETH perpetual data** (2020-01 → 2024-06, sealed period untouched): 39,409 bars
and 4,927 funding settlements per symbol, all passing the §6 gate with zero findings. Every §14.2
check was re-run against real prices, including the funding sign test against 4,926 actual
settlements.

Real data exposed four engine defects that 230 passing tests had not — most seriously, a
`max_gross_leverage` limit that was declared but never checked against realised exposure, which
combined with capacity-rejected delevering to run a book to 32,000× leverage. All four are fixed
and covered by regression tests. The report documents them.

**Note on sources:** `fapi.binance.com` returns HTTP 451 (geo-restricted) from some hosts, for every
endpoint. Funding therefore defaults to the monthly archive, which is the better source regardless.
Open interest has no archive fallback and needs a host Binance will serve.

## Constraints you cannot engineer around

**Archive coverage starts 2020-01.** The §10.1 train window opens 2019-01-01, and the monthly
UM-futures archive has no months before 2020-01 for BTCUSDT or ETHUSDT. The first year of the train
window cannot be filled from this source.

**Open interest.** Binance retains ~30 days of open-interest history and it cannot be backfilled at
any price. Every day the daily collector does not run is a day permanently lost. `cryptolab init`
prints the concrete cut-off date. OI-based hypotheses are untestable on history until the collector
has been running for a while.

**Liquidations.** The public feed is partial and throttled. Any liquidation-derived series carries a
lower confidence flag and may not be used in a promotable strategy.

**Contract specs change.** Tick size, multiplier and funding interval all move. The funding interval
is inferred from the data rather than assumed to be 8h.

**Basis uses mark price**, never last-traded price.

## The tests that matter

`tests/test_antilookahead.py` implements SPEC.md §14.2 and runs in CI on every push. The shift test
and the shuffle test are the two that catch real bugs, so each is also asserted against
`LookaheadSignal` — a deliberately broken strategy that peeks at the next bar. If those two
true-positive tests ever pass, the suite is broken, not the signal.

The shuffle test asserts on **gross** returns. Run on net returns it would measure the cost drag
instead of foresight, and would "pass" any strategy that merely trades enough to lose money.

## Layout

```
src/cryptolab/
  data/        sources/, schemas.py, ingest.py, quality.py, store.py
  features/    returns, volatility, derivatives, registry
  signals/     base.py (the Signal ABC) — no Tier-1 signals yet
  backtest/    engine.py, costs.py, portfolio.py, risk.py
  validation/  walkforward, deflated_sharpe, pbo, registry, gates, sealed, synthetic
  cli.py
```
