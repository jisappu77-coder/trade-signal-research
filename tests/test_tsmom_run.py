"""The Phase 4 runner: registration discipline, fold slicing, and per-bar statistics."""

from __future__ import annotations

import polars as pl
import pytest

from cryptolab.data.store import ParquetStore
from cryptolab.reporting.report import write_site
from cryptolab.reporting.tsmom_run import STRATEGY_FAMILY, build_reports, run_grid
from cryptolab.validation.registry import TrialRegistry
from cryptolab.validation.synthetic import synthetic_bars

# ~2.5 years of hourly bars from 2020-01-01, deep enough not to trip the §7 capacity limit.
BARS = 22_000
START, END = "2020-01-01", "2022-06-30"


@pytest.fixture(scope="module")
def lake(tmp_path_factory, module_splits):
    """Module-scoped: running the grid is expensive and every test here wants the same lake."""
    tmp_path = tmp_path_factory.mktemp("lake")
    splits = module_splits
    registry = TrialRegistry(tmp_path / "registry.sqlite")
    store = registry.bind_store(ParquetStore(tmp_path / "data", splits))
    frame = synthetic_bars(BARS, seed=5, start_ms=1_577_836_800_000, quote_volume=5e9)
    settlements = frame["open_time"].to_list()[::8]
    for symbol in ("BTCUSDT", "ETHUSDT"):
        store.write(frame, "ohlcv", exchange="binance", symbol=symbol, source_uri="test://")
        store.write(
            pl.DataFrame(
                {
                    "funding_time": settlements,
                    "symbol": [symbol] * len(settlements),
                    "funding_rate": [0.0001] * len(settlements),
                    "interval_hours": [8.0] * len(settlements),
                    "mark_price": [20_000.0] * len(settlements),
                }
            ),
            "funding",
            exchange="binance",
            symbol=symbol,
            source_uri="test://",
        )
    yield store, registry
    registry.close()


@pytest.fixture(scope="module")
def grid(lake):
    """The 4h grid, run once. Re-running it per test dominated the suite's runtime."""
    store, registry = lake
    return run_grid(store, registry, start=START, end=END, bars_filter=["4h"])


@pytest.fixture(scope="module")
def reports(lake):
    store, registry = lake
    return build_reports(store, registry, start=START, end=END, bars_filter=["4h"])


# ---- registration discipline (§10.4, §8.1) ------------------------------------------


def test_the_whole_declared_space_is_registered_even_when_a_subset_runs(grid):
    """Running only 4h must not understate N — the search space is what was declared."""
    runs, n_trials = grid
    assert len(runs) == 24  # 12 combinations x 2 symbols actually executed
    assert n_trials == 48  # but all 24 x 2 declared combinations registered


def test_n_is_read_from_the_registry_not_hard_coded(lake, grid):
    _, registry = lake
    assert grid[1] == registry.count(signal="tsmom")


def test_rerunning_does_not_inflate_n(lake, grid):
    """Recomputing is not a new statistical trial; only a new parameter tuple is."""
    store, registry = lake
    _, second = run_grid(store, registry, start=START, end=END, bars_filter=["4h"])
    assert second == grid[1] == 48


def test_trials_carry_the_strategy_family(lake, grid):
    """The family is what the one-shot sealed-test token is bound to, so it must be explicit."""
    _, registry = lake
    assert all(t.strategy_family == STRATEGY_FAMILY for t in registry.all_trials())


def test_the_registry_chain_stays_intact(lake, grid):
    _, registry = lake
    assert registry.verify_chain()


def test_no_sealed_token_is_ever_issued(lake, grid):
    """§10.1 allows one opening per family, ever. Phase 4 must not spend it."""
    _, registry = lake
    # If Phase 4 had requested a token, issuing one now would raise.
    registry.issue_test_token(STRATEGY_FAMILY)


def test_an_empty_lake_is_refused(tmp_path, splits):
    registry = TrialRegistry(tmp_path / "r.sqlite")
    store = registry.bind_store(ParquetStore(tmp_path / "data", splits))
    with pytest.raises(ValueError, match="ingest the data first"):
        run_grid(store, registry, start=START, end=END)
    registry.close()


# ---- fold handling ------------------------------------------------------------------


def test_folds_are_slices_not_cold_restarts(grid):
    """Targets are computed once on the full series; a fold that regenerated would warm up cold."""
    run = grid[0][0]
    assert run.fold_sharpes, "expected at least one evaluated fold"
    # A cold-started 4h fold would spend a third of its window flat and score exactly zero.
    assert any(s != 0.0 for s in run.fold_sharpes)


def test_every_run_carries_its_attribution(grid):
    for run in grid[0]:
        assert run.attribution.residual == pytest.approx(0.0, abs=1e-6)


# ---- reports ------------------------------------------------------------------------


def test_reports_are_tier_one_and_promotable(reports):
    assert len(reports) == 24
    assert all(r.tier == 1 and r.promotable for r in reports)


def test_reports_carry_leg_attribution_and_tax(reports):
    """§8.1 and §17 both require these in the report, not only in the log."""
    report = reports[0]
    assert {leg["leg"] for leg in report.legs} == {"long", "short", "flat"}
    assert report.post_tax_return is not None
    assert report.pre_tax_return is not None
    assert "pre-tax" in report.tax_line


def test_verdicts_are_computed_from_the_gates(reports):
    for report in reports:
        assert report.verdict_line == report.gates.verdict_line()


def test_site_writes_an_index_and_a_page_per_run(reports, tmp_path):
    paths = write_site(reports, tmp_path / "site")
    assert paths[0].name == "index.html"
    assert len(paths) == 25  # index + 24 runs
    assert "N = 48" in (tmp_path / "site" / paths[1].name).read_text()
