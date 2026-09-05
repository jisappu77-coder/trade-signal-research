"""The Phase 6b runner: registration discipline, the store round trip, and the results record."""

from __future__ import annotations

import json

import polars as pl
import pytest

from cryptolab.backtest.carry_portfolio import load_aligned, load_symbol, run_portfolio, with_capacity
from cryptolab.data.store import ParquetStore
from cryptolab.reporting.carry_universe_run import (
    STRATEGY_FAMILY,
    declared_grid,
    run_universe_grid,
    summarise,
    write_results,
)
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.synthetic import synthetic_bars
from cryptolab.validation.tax import fixed_deposit_hurdle_apr, tax_single_run

BARS = 6_000
START, END = "2020-01-01", "2020-09-30"
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")


@pytest.fixture(scope="module")
def lake(tmp_path_factory, module_splits):
    """A small three-symbol lake with funding high enough that the sleeve actually deploys."""
    root = tmp_path_factory.mktemp("universe")
    store = ParquetStore(root / "data", module_splits)
    for index, symbol in enumerate(SYMBOLS):
        frame = synthetic_bars(BARS, seed=11 + index, start_ms=1_577_836_800_000, quote_volume=5e9)
        settlements = frame["open_time"].to_list()[::8]
        store.write(frame, "ohlcv", exchange="binance", symbol=symbol, source_uri="test://")
        store.write(frame, "spot_ohlcv", exchange="binance", symbol=symbol, source_uri="test://")
        store.write(
            pl.DataFrame(
                {
                    "funding_time": settlements,
                    "symbol": [symbol] * len(settlements),
                    # Well above every entry threshold in the grid, so episodes exist to measure.
                    "funding_rate": [0.01] * len(settlements),
                    "interval_hours": [8.0] * len(settlements),
                    "mark_price": [None] * len(settlements),
                },
                schema_overrides={"mark_price": pl.Float64},
            ),
            "funding",
            exchange="binance",
            symbol=symbol,
            source_uri="test://",
        )
    return root, store


# ---- loading ------------------------------------------------------------------------


def test_a_symbol_with_every_leg_loads(lake):
    _, store = lake
    assert load_aligned(store, SYMBOLS[0], START, END) is not None


def test_a_symbol_missing_a_leg_is_none_not_an_error(lake):
    """Two legs cannot be carried against one; the symbol simply is not tradeable."""
    _, store = lake
    assert load_aligned(store, "NOSUCHUSDT", START, END) is None


def test_load_symbol_marks_capacity(lake):
    _, store = lake
    data = load_symbol(store, SYMBOLS[0], START, END, capital=25_000.0, max_positions=8)
    assert data is not None
    assert data.entry_allowed.shape[0] == data.aligned.height


def test_load_symbol_is_none_for_a_missing_symbol(lake):
    _, store = lake
    assert load_symbol(store, "NOSUCHUSDT", START, END) is None


def test_a_smaller_slot_can_be_filled_where_a_larger_one_cannot(lake):
    """The participation limit is what excludes a market, so slot size decides eligibility."""
    _, store = lake
    loaded = load_aligned(store, SYMBOLS[0], START, END)
    assert loaded is not None
    small = with_capacity(*loaded, capital=25_000.0, max_positions=12)
    large = with_capacity(*loaded, capital=250_000_000.0, max_positions=1)
    assert small.entry_allowed.sum() > large.entry_allowed.sum()


def test_funding_apr_is_exposed_per_bar(lake):
    _, store = lake
    data = load_symbol(store, SYMBOLS[0], START, END)
    assert data is not None
    assert data.funding_apr.shape[0] == data.aligned.height


# ---- the declared search ------------------------------------------------------------


def test_the_grid_is_the_full_product_of_its_axes():
    grid = declared_grid()
    assert len(grid) == 81
    assert len({json.dumps(p, sort_keys=True) for p in grid}) == 81


def test_the_grid_carries_no_cost_axis():
    """§18: the cost model is not a knob. No configuration may reach it."""
    forbidden = {"taker_fee_bps", "slippage_bps", "regime", "impact_multiplier", "half_spread_bps"}
    assert all(forbidden.isdisjoint(params) for params in declared_grid())


# ---- the run ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def executed(lake, tmp_path_factory):
    root, store = lake
    with TrialRegistry(root / "registry.sqlite") as registry:
        runs, n, used = run_universe_grid(
            store, registry, list(SYMBOLS), start=START, end=END, capital=25_000.0
        )
        return runs, n, used, root


def test_every_declared_configuration_is_run(executed):
    runs, _, _, _ = executed
    assert len(runs) == len(declared_grid())


def test_n_counts_the_whole_declared_search(executed):
    """Widening a search raises `N`, which lowers every deflated Sharpe drawn from it."""
    _, n, _, _ = executed
    assert n == len(declared_grid())


def test_rerunning_does_not_inflate_n(lake):
    root, store = lake
    with TrialRegistry(root / "rerun.sqlite") as registry:
        _, first, _ = run_universe_grid(store, registry, list(SYMBOLS)[:1], start=START, end=END)
        _, second, _ = run_universe_grid(store, registry, list(SYMBOLS)[:1], start=START, end=END)
    assert first == second


def test_trials_carry_the_carry_family(executed):
    """Phase 6b must deflate against Phase 6's search, not start a fresh count."""
    _, _, _, root = executed
    with TrialRegistry(root / "registry.sqlite") as registry:
        assert registry.count(strategy_family=STRATEGY_FAMILY) > 0
        assert all(t.strategy_family == STRATEGY_FAMILY for t in registry.all_trials())


def test_the_registry_chain_stays_intact(executed):
    _, _, _, root = executed
    with TrialRegistry(root / "registry.sqlite") as registry:
        assert registry.verify_chain()


def test_no_sealed_token_is_ever_issued(executed):
    _, _, _, root = executed
    with TrialRegistry(root / "registry.sqlite") as registry:
        assert all("token" not in t.note.lower() for t in registry.all_trials())


def test_symbols_used_are_reported(executed):
    _, _, used, _ = executed
    assert used == sorted(SYMBOLS)


def test_a_run_reports_both_tax_treatments(executed):
    """§17: pre- and post-tax are separate numbers, never one blended figure."""
    runs, _, _, _ = executed
    profitable = [r for r in runs if r.pre_tax_apr > 0]
    assert profitable
    assert all(r.post_tax_apr < r.pre_tax_apr for r in profitable)


def test_a_losing_run_gets_no_tax_relief(executed):
    """§115BBH's asymmetry: a loss is not offset, so post-tax equals pre-tax."""
    runs, _, _, _ = executed
    for run in runs:
        if run.pre_tax_apr < 0:
            assert run.post_tax_apr == pytest.approx(run.pre_tax_apr)


def test_the_fixed_deposit_comparison_is_post_tax_on_both_sides(executed):
    """The original error: a 30%-taxed return compared against an *untaxed* deposit rate."""
    runs, _, _, _ = executed
    for run in runs:
        assert run.beats_fixed_deposit(0.07, 0.30) == (run.post_tax_apr > fixed_deposit_hurdle_apr())


def test_a_higher_slab_lowers_the_bar(executed):
    """A deposit is worth less to a top-bracket holder, so the hurdle it sets is lower."""
    runs, _, _, _ = executed
    assert fixed_deposit_hurdle_apr(0.07, 0.30) < fixed_deposit_hurdle_apr(0.07, 0.05)
    # A run can clear the top-slab hurdle and miss the zero-slab one; never the reverse.
    for run in runs:
        assert run.beats_fixed_deposit(0.07, 0.0) <= run.beats_fixed_deposit(0.07, 0.30)


def test_only_a_zero_slab_holder_keeps_the_headline_rate():
    assert fixed_deposit_hurdle_apr(0.07, 0.0) == pytest.approx(0.07)
    assert fixed_deposit_hurdle_apr(0.07, 0.30) == pytest.approx(0.07 * (1 - 0.312))


def test_an_impossible_slab_is_refused():
    with pytest.raises(ValueError, match="slab_rate"):
        fixed_deposit_hurdle_apr(0.07, 1.5)


def test_cess_is_charged_on_vda_gains():
    """§17 names 30%; the cess is levied on top and omitting it flatters every post-tax figure."""
    outcome = tax_single_run(pre_tax_pnl=1_000.0, traded_notional=0.0, initial_equity=25_000.0, years=1.0)
    assert outcome.tax_due == pytest.approx(1_000.0 * 0.312)
    assert outcome.post_tax_pnl == pytest.approx(688.0)


def test_a_loss_still_gets_no_relief_under_cess():
    """§115BBH's asymmetry is untouched by the refinement: a loss buys nothing."""
    outcome = tax_single_run(pre_tax_pnl=-500.0, traded_notional=0.0, initial_equity=25_000.0, years=1.0)
    assert outcome.tax_due == 0.0
    assert outcome.post_tax_pnl == pytest.approx(-500.0)


def test_liquidation_rate_is_reported_beside_the_return(executed):
    """A return without its liquidation column misleads; both must exist on every run."""
    runs, _, _, _ = executed
    for run in runs:
        assert 0.0 <= run.liquidation_rate <= 1.0
        if run.episodes == 0:
            assert run.liquidation_rate == 0.0


def test_summarise_handles_a_run_that_never_deployed(lake):
    _, store = lake
    data = load_symbol(store, SYMBOLS[0], START, END)
    assert data is not None
    result = run_portfolio([data], entry_threshold_apr=99.0, exit_threshold_apr=0.0)
    params = {"min_holding_days": 7.0, "exit_fraction": 0.0, "max_positions": 8, "margin_rate": 0.2}
    run = summarise(result, params, 99.0, 25_000.0)
    assert run.episodes == 0
    assert run.hit_rate == 0.0
    assert run.post_tax_apr == 0.0


# ---- the record ---------------------------------------------------------------------


def test_every_run_is_written_including_the_losers(executed, tmp_path):
    """§12: the failure record is the product, so nothing is filtered on the way out."""
    runs, n, used, _ = executed
    path = write_results(runs, n, used, tmp_path / "results.json")
    payload = json.loads(path.read_text())
    assert len(payload["runs"]) == len(runs)
    assert payload["trials_n"] == n
    assert payload["symbols"] == used


def test_the_record_carries_the_trial_count(executed, tmp_path):
    """A result without its `N` cannot be deflated, so the two travel together."""
    runs, n, used, _ = executed
    payload = json.loads(write_results(runs, n, used, tmp_path / "r.json").read_text())
    assert payload["trials_n"] == n
