# CLAUDE.md — working rules for `cryptolab`

Read `SPEC.md` in full before writing code. This file governs *how* you work; SPEC.md governs *what* you build.

## The one-line summary
You are building an instrument that tries to **disprove** the existence of a trading edge. Success is a
trustworthy verdict, not a good-looking equity curve.

## Order of work
Follow the phases in SPEC.md §15 strictly. **Phase 3 (validation harness) must be green before Phase 4 (first
signal).** If you find yourself wanting to "just quickly test a signal idea" before the harness exists, stop —
that impulse is exactly what the phase ordering is defending against.

## Non-negotiables
1. Never report a Sharpe ratio without its deflated Sharpe and the trial count `N` it was deflated by.
2. Never let a backtest fill inside a bar. Signals at bar `t` close, fills at bar `t+1` open.
3. Never make the cost model configurable downward to make a result look better. The four regimes in
   SPEC.md §7 are fixed.
4. Never touch the sealed test period except through the one-time token mechanism.
5. Never delete a failed strategy report. The failure record is the product.

## When a strategy fails a gate
Report the failure. Write the verdict line. Move on. Do **not**:
- widen the parameter grid and re-run
- change the cost regime
- shorten the test period to a friendlier window
- add a filter that happens to exclude the losing trades

If you genuinely believe a parameter range was mis-specified, say so explicitly, register the expanded search
as new trials, and note in the report that `N` increased — which will lower the deflated Sharpe accordingly.
That is the honest cost of a second look.

## Code conventions
- `mypy --strict` on `src/`, `ruff` clean, no `# type: ignore` without a comment explaining why
- Polars over pandas; no `.apply()` in hot paths
- All timestamps int64 UTC ms, bar-open convention, named `*_time` or `timestamp`
- No global mutable state; no module-level I/O
- Every public function that returns a metric returns it with units in the name (`_bps`, `_apr`, `_annualised`)
- Notebooks in `research/` may import from `src/`; `src/` may never import from `research/`

## Testing discipline
Write the anti-lookahead tests (SPEC.md §14.2) **before** the first signal, not after. The shuffle test and
the shift test are the two that catch real bugs; run them in CI on every signal.

## Data caveats you must handle, not assume away
- Binance open-interest history is only retrievable for ~30 days. Start the daily collector in Phase 1 or
  accept that OI-based hypotheses are permanently untestable on historical data.
- The public liquidation websocket feed is partial and throttled. Any liquidation-derived series carries a
  lower confidence flag and may not be used in a promotable strategy.
- Contract specifications change. Refuse to backtest across an unhandled spec change.
- Use mark price for basis, never last-traded price.

## Communication style
When reporting results to the user, lead with the verdict and the break-even cost, not the equity curve.
State plainly when an edge does not survive costs. Hedged language about "promising early results" on a
strategy with deflated Sharpe below the gate is a failure of this document.

## Scope discipline
v1 places no orders and holds no trade-permissioned API keys. If asked to add live execution, first confirm
that at least one strategy has passed all §11 gates, and then treat live wiring as a separate project with its
own spec and its own risk review.
