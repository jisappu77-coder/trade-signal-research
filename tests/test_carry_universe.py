"""The universe expansion: discovery, cross-sectional selection, and the portfolio run.

Network is mocked throughout — the suite never reaches the archive.
"""

from __future__ import annotations

import asyncio
import io
import zipfile

import httpx
import numpy as np
import polars as pl
import pytest

from cryptolab.backtest.carry import _leg_costs, align_legs, funding_apr, run_carry_backtest
from cryptolab.backtest.carry_portfolio import (
    PortfolioResult,
    SymbolData,
    build_panels,
    run_portfolio,
)
from cryptolab.backtest.costs import MAX_PARTICIPATION, fill_cost, get_regime
from cryptolab.data.sources import binance_archive
from cryptolab.data.universe import (
    SymbolCoverage,
    discover_symbols,
    list_archive_symbols,
    probe_coverage,
    tradeable,
)
from cryptolab.signals.carry_xs import SymbolPanel, select

HOUR = 3_600_000
REGIME = get_regime("conservative")


# ---- universe discovery -------------------------------------------------------------


def listing(prefixes: list[str], *, truncated: bool = False, next_marker: str = "") -> str:
    body = "".join(f"<Prefix>{p}</Prefix>" for p in prefixes)
    tail = f"<NextMarker>{next_marker}</NextMarker>" if truncated else ""
    return (
        f"<ListBucketResult><IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        f"{tail}{body}</ListBucketResult>"
    )


def test_listing_pages_through_a_truncated_response():
    """The bucket caps a listing at 1000 keys; a single page would silently truncate the universe."""
    prefix = "data/futures/um/monthly/klines/"
    pages = [
        listing([f"{prefix}AAAUSDT/"], truncated=True, next_marker=f"{prefix}AAAUSDT/"),
        listing([f"{prefix}BBBUSDT/"]),
    ]
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        page = pages[calls["n"]]
        calls["n"] += 1
        return httpx.Response(200, text=page)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    assert asyncio.run(list_archive_symbols(client, prefix)) == ["AAAUSDT", "BBBUSDT"]
    assert calls["n"] == 2


def test_discovery_keeps_only_symbols_with_both_legs():
    """A perp without a spot pair cannot be carried, however liquid it is."""

    def handle(request: httpx.Request) -> httpx.Response:
        prefix = request.url.params["prefix"]
        names = ["BTCUSDT", "ETHUSDT", "PERPONLYUSDT"] if "futures" in prefix else ["BTCUSDT", "ETHUSDT"]
        return httpx.Response(200, text=listing([f"{prefix}{n}/" for n in names]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    assert asyncio.run(discover_symbols(client)) == ["BTCUSDT", "ETHUSDT"]


def test_discovery_filters_by_quote_currency():
    def handle(request: httpx.Request) -> httpx.Response:
        prefix = request.url.params["prefix"]
        return httpx.Response(200, text=listing([f"{prefix}{n}/" for n in ("BTCUSDT", "BTCUSDC")]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    assert asyncio.run(discover_symbols(client, quote="USDT")) == ["BTCUSDT"]


def test_a_month_needs_all_three_legs():
    """Perp and spot without funding is not a carry month, and must not be counted as one."""

    def handle(request: httpx.Request) -> httpx.Response:
        missing_funding = "fundingRate" in request.url.path and "NOFUNDUSDT" in request.url.path
        return httpx.Response(404 if missing_funding else 200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    coverage = asyncio.run(probe_coverage(client, ["OKUSDT", "NOFUNDUSDT"], [(2022, 6)]))
    got = {entry.symbol: entry.listed for entry in coverage}
    assert got == {"OKUSDT": True, "NOFUNDUSDT": False}


def test_a_transport_failure_is_retried_not_read_as_absent():
    """Reading a network blip as "did not exist" would quietly shrink the universe."""
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    coverage = asyncio.run(probe_coverage(client, ["BTCUSDT"], [(2022, 6)]))
    assert coverage[0].listed


def test_tradeable_keeps_delisted_symbols():
    """The whole point: a symbol that traded in 2021 and died belongs in the universe."""
    coverage = [
        SymbolCoverage("LUNAUSDT", ((2021, 6),)),
        SymbolCoverage("BTCUSDT", ((2021, 6), (2023, 6))),
        SymbolCoverage("NEVERUSDT", ()),
    ]
    assert tradeable(coverage) == ["BTCUSDT", "LUNAUSDT"]


# ---- per-bar costs ------------------------------------------------------------------


def test_vectorised_leg_cost_equals_fill_cost():
    """The fast path must be the same arithmetic as §7's reference implementation, not a lookalike."""
    volumes = np.array([3.4e8, 5.0e6, 2.0e6, 8.0e5])
    notional = 16_666.0
    cost, _ = _leg_costs(notional, volumes, volumes, REGIME, half_spread_bps=1.0)
    for i, volume in enumerate(volumes):
        reference = fill_cost(
            notional, float(volume), REGIME, half_spread_bps=1.0, max_participation=1.0
        ).total
        assert cost[i] == pytest.approx(reference, rel=1e-12)


def test_a_thin_market_is_refused_entry_rather_than_priced_as_deep():
    notional = 16_666.0
    thin = notional / MAX_PARTICIPATION * 0.5  # twice the participation limit
    deep = np.array([3.4e8])
    _, allowed_deep = _leg_costs(notional, deep, deep, REGIME, half_spread_bps=1.0)
    _, allowed_thin = _leg_costs(notional, np.array([thin]), np.array([thin]), REGIME, half_spread_bps=1.0)
    assert allowed_deep[0]
    assert not allowed_thin[0]


def test_a_refused_bar_is_still_priced_so_a_forced_exit_is_not_free():
    notional = 16_666.0
    thin = np.array([notional / MAX_PARTICIPATION * 0.5])
    cost, allowed = _leg_costs(notional, thin, thin, REGIME, half_spread_bps=1.0)
    assert not allowed[0]
    assert cost[0] > 0.0


def test_a_thinner_market_never_costs_less():
    volumes = np.array([1e9, 1e8, 1e7, 1e6])
    cost, _ = _leg_costs(16_666.0, volumes, volumes, REGIME, half_spread_bps=1.0)
    assert np.all(np.diff(cost) > 0)


def test_a_zero_volume_bar_is_refused_and_capped():
    """A dead bar has unbounded modelled impact; it must not become an infinite charge."""
    volumes = np.array([0.0])
    cost, allowed = _leg_costs(16_666.0, volumes, volumes, REGIME, half_spread_bps=1.0)
    assert not allowed[0]
    assert np.isfinite(cost[0])


# ---- cross-sectional selection ------------------------------------------------------


def panel(symbol: str, apr: list[float], allowed: list[bool] | None = None) -> SymbolPanel:
    return SymbolPanel(
        symbol=symbol,
        funding_apr=np.array(apr, dtype=float),
        entry_allowed=np.array([True] * len(apr) if allowed is None else allowed, dtype=bool),
    )


def test_no_more_than_max_positions_are_ever_held():
    """The whole additivity argument rests on this; if it fails the portfolio is levered."""
    panels = [panel(f"S{i}", [1.0] * 10) for i in range(6)]
    targets = select(panels, entry_threshold_apr=0.3, exit_threshold_apr=0.0, max_positions=2)
    occupancy = np.sum(np.vstack(list(targets.values())), axis=0)
    assert occupancy.max() <= 2


def test_the_highest_paying_eligible_symbol_wins_a_free_slot():
    panels = [panel("LOW", [0.4] * 3), panel("HIGH", [0.9] * 3)]
    targets = select(panels, entry_threshold_apr=0.3, exit_threshold_apr=0.0, max_positions=1)
    assert targets["HIGH"][0] == 1.0
    assert targets["LOW"][0] == 0.0


def test_an_incumbent_is_not_evicted_by_a_better_candidate():
    """Re-ranking every bar would pay four fills to chase a few basis points."""
    panels = [panel("IN", [0.9, 0.5, 0.5]), panel("BETTER", [0.0, 2.0, 2.0])]
    targets = select(panels, entry_threshold_apr=0.4, exit_threshold_apr=0.1, max_positions=1)
    assert list(targets["IN"]) == [1.0, 1.0, 1.0]
    assert list(targets["BETTER"]) == [0.0, 0.0, 0.0]


def test_an_incumbent_leaves_when_its_own_exit_threshold_is_crossed():
    panels = [panel("IN", [0.9, 0.05])]
    targets = select(panels, entry_threshold_apr=0.4, exit_threshold_apr=0.1, max_positions=1)
    assert list(targets["IN"]) == [1.0, 0.0]


def test_a_freed_slot_is_reused_the_same_bar():
    panels = [panel("OUT", [0.9, 0.0]), panel("NEXT", [0.5, 0.9])]
    targets = select(panels, entry_threshold_apr=0.4, exit_threshold_apr=0.1, max_positions=1)
    assert list(targets["OUT"]) == [1.0, 0.0]
    assert list(targets["NEXT"]) == [0.0, 1.0]


def test_a_symbol_with_no_data_is_never_selected():
    """`nan` is "this market did not exist", not "funding was zero"."""
    panels = [panel("DEAD", [np.nan, np.nan]), panel("LIVE", [0.9, 0.9])]
    targets = select(panels, entry_threshold_apr=0.4, exit_threshold_apr=0.1, max_positions=4)
    assert list(targets["DEAD"]) == [0.0, 0.0]
    assert list(targets["LIVE"]) == [1.0, 1.0]


def test_a_delisted_incumbent_gives_up_its_slot():
    panels = [panel("DIES", [0.9, np.nan, np.nan]), panel("OTHER", [0.5, 0.5, 0.5])]
    targets = select(panels, entry_threshold_apr=0.4, exit_threshold_apr=0.1, max_positions=1)
    assert list(targets["DIES"]) == [1.0, 0.0, 0.0]
    # The slot the delisting freed is taken the same bar, and held while OTHER clears its own exit.
    assert list(targets["OTHER"]) == [0.0, 1.0, 1.0]


def test_a_market_too_thin_to_enter_is_skipped_even_when_it_pays_most():
    """The highest funding is often in the market least able to absorb the order."""
    panels = [panel("THIN", [5.0], allowed=[False]), panel("DEEP", [0.9])]
    targets = select(panels, entry_threshold_apr=0.4, exit_threshold_apr=0.1, max_positions=2)
    assert targets["THIN"][0] == 0.0
    assert targets["DEEP"][0] == 1.0


def test_selection_is_causal():
    """Truncating the input must not change any decision that was already made."""
    apr = [0.9, 0.2, 0.8, 0.1, 0.7, 0.95]
    full = select([panel("S", apr)], entry_threshold_apr=0.5, exit_threshold_apr=0.3, max_positions=1)
    for cut in range(1, len(apr)):
        part = select(
            [panel("S", apr[:cut])], entry_threshold_apr=0.5, exit_threshold_apr=0.3, max_positions=1
        )
        assert list(part["S"]) == list(full["S"][:cut]), f"decisions changed at truncation {cut}"


def test_an_exit_threshold_above_entry_is_refused():
    with pytest.raises(ValueError, match="deploy and unwind"):
        select([panel("S", [1.0])], entry_threshold_apr=0.2, exit_threshold_apr=0.5)


def test_panels_must_share_a_timeline():
    with pytest.raises(ValueError, match="master timeline"):
        select([panel("A", [1.0, 1.0]), panel("B", [1.0])], entry_threshold_apr=0.2, exit_threshold_apr=0.1)


def test_zero_slots_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        select([panel("S", [1.0])], entry_threshold_apr=0.2, exit_threshold_apr=0.1, max_positions=0)


# ---- the portfolio run --------------------------------------------------------------


def synthetic(symbol: str, n: int, *, rate: float, start: int = 0, volume: float = 1e9) -> SymbolData:
    times = np.arange(n, dtype=np.int64) * HOUR + start
    price = np.full(n, 100.0)
    aligned = pl.DataFrame(
        {
            "open_time": times,
            "perp_open": price,
            "perp_close": price,
            "perp_quote_volume": np.full(n, volume),
            "spot_open": price,
            "spot_close": price,
            "spot_quote_volume": np.full(n, volume),
            "funding_in_force": np.full(n, rate),
            "settlement_rate": np.where(np.arange(n) % 8 == 0, rate, 0.0),
        }
    )
    return SymbolData(symbol=symbol, aligned=aligned, entry_allowed=np.ones(n, dtype=bool), dropped_bars=0)


def test_panels_use_the_union_of_timelines_not_the_intersection():
    """An intersection would impose the youngest symbol's window on the whole run."""
    old = synthetic("OLD", 10, rate=0.001)
    new = synthetic("NEW", 5, rate=0.001, start=5 * HOUR)
    master, panels = build_panels([old, new])
    assert len(master) == 10
    by_symbol = {p.symbol: p for p in panels}
    assert np.isnan(by_symbol["NEW"].funding_apr[:5]).all()
    assert np.isfinite(by_symbol["NEW"].funding_apr[5:]).all()


def test_a_late_listing_contributes_only_its_own_bars():
    late = synthetic("LATE", 4, rate=0.001, start=6 * HOUR)
    _, panels = build_panels([synthetic("EARLY", 10, rate=0.001), late])
    assert int(np.isfinite(next(p for p in panels if p.symbol == "LATE").funding_apr).sum()) == 4


def test_the_portfolio_is_the_sum_of_its_slots():
    """The additivity claim the whole design rests on, checked rather than asserted in prose."""
    loaded = [synthetic(f"S{i}", 200, rate=0.005) for i in range(3)]
    result = run_portfolio(
        loaded, entry_threshold_apr=0.3, exit_threshold_apr=0.0, capital=24_000.0, max_positions=3
    )
    assert result.net_pnl == pytest.approx(sum(r.net_pnl for r in result.per_symbol.values()))

    slot = 24_000.0 / 3
    for symbol, sub in result.per_symbol.items():
        alone = run_carry_backtest(
            next(d for d in loaded if d.symbol == symbol).aligned,
            pl.DataFrame(
                {
                    "timestamp": sub.equity_curve["open_time"],
                    "target_position": np.ones(sub.equity_curve.height),
                    "confidence": np.ones(sub.equity_curve.height),
                }
            ),
            initial_equity=slot,
        )
        assert sub.net_pnl == pytest.approx(alone.net_pnl, rel=1e-9)


def test_occupancy_never_exceeds_the_slot_count():
    loaded = [synthetic(f"S{i}", 100, rate=0.005) for i in range(6)]
    result = run_portfolio(loaded, entry_threshold_apr=0.3, exit_threshold_apr=0.0, max_positions=2)
    assert result.occupancy.max() <= 2
    assert result.deployment_fraction <= 1.0


def test_apr_counts_idle_slots():
    """Reporting return on deployed capital would flatter a sleeve that is mostly in cash."""
    loaded = [synthetic("ONLY", 8760, rate=0.005)]
    result = run_portfolio(
        loaded, entry_threshold_apr=0.3, exit_threshold_apr=0.0, capital=25_000.0, max_positions=5
    )
    assert result.deployment_fraction == pytest.approx(0.2, abs=1e-6)
    assert result.apr < result.net_pnl / (25_000.0 / 5) / result.span_years


def test_an_empty_universe_is_an_empty_result_not_a_crash():
    result = run_portfolio([], entry_threshold_apr=0.3, exit_threshold_apr=0.0)
    assert isinstance(result, PortfolioResult)
    assert result.net_pnl == 0.0
    assert result.episodes == []


def test_a_symbol_that_never_fires_is_not_run():
    loaded = [synthetic("HIGH", 100, rate=0.005), synthetic("LOW", 100, rate=0.00001)]
    result = run_portfolio(loaded, entry_threshold_apr=0.3, exit_threshold_apr=0.0, max_positions=4)
    assert "LOW" not in result.per_symbol
    assert "HIGH" in result.per_symbol


def test_align_legs_carries_spot_volume_for_the_cost_model():
    """Without it the spot leg would be priced against the perp's depth."""
    n = 5
    times = np.arange(n, dtype=np.int64) * HOUR
    frame = pl.DataFrame(
        {
            "open_time": times,
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1.0),
            "quote_volume": np.full(n, 1e8),
            "trades": np.ones(n, dtype=np.int64),
            "close_time": times + HOUR - 1,
        }
    )
    funding = pl.DataFrame(
        {"funding_time": times[:1], "funding_rate": [0.001], "interval_hours": [8.0], "symbol": ["X"]}
    )
    aligned, _ = align_legs(frame, frame.with_columns(pl.col("quote_volume") * 0.5), funding)
    assert "spot_quote_volume" in aligned.columns
    assert aligned["spot_quote_volume"][0] == pytest.approx(5e7)


# ---- the funding cadence ------------------------------------------------------------


def funding_frame(times: np.ndarray, rate: float, interval: float | list[float]) -> pl.DataFrame:
    hours = [interval] * len(times) if isinstance(interval, float) else interval
    return pl.DataFrame(
        {
            "funding_time": times,
            "symbol": ["X"] * len(times),
            "funding_rate": [rate] * len(times),
            "interval_hours": hours,
            "mark_price": [None] * len(times),
        },
        schema_overrides={"mark_price": pl.Float64},
    )


def bars(n: int) -> pl.DataFrame:
    times = np.arange(n, dtype=np.int64) * HOUR
    return pl.DataFrame(
        {
            "open_time": times,
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1.0),
            "quote_volume": np.full(n, 1e8),
            "trades": np.ones(n, dtype=np.int64),
            "close_time": times + HOUR - 1,
        }
    )


def test_a_four_hour_symbol_pays_twice_what_an_eight_hour_one_does():
    """The same per-settlement rate is twice the APR when it settles twice as often."""

    times = np.arange(0, 5, dtype=np.int64) * HOUR
    frame = bars(5)
    eight, _ = align_legs(frame, frame, funding_frame(times, 0.0001, 8.0))
    four, _ = align_legs(frame, frame, funding_frame(times, 0.0001, 4.0))
    assert funding_apr(eight)[-1] == pytest.approx(0.1095, rel=1e-6)
    assert funding_apr(four)[-1] == pytest.approx(0.2190, rel=1e-6)


def test_a_mid_stream_cadence_change_is_followed_not_averaged():
    """A venue shortens the interval when funding runs extreme; the APR must follow it."""

    times = np.arange(0, 4, dtype=np.int64) * HOUR
    frame = bars(4)
    aligned, _ = align_legs(frame, frame, funding_frame(times, 0.0001, [8.0, 8.0, 4.0, 4.0]))
    apr = funding_apr(aligned)
    assert apr[1] == pytest.approx(0.1095, rel=1e-6)
    assert apr[3] == pytest.approx(0.2190, rel=1e-6)


def test_the_cadence_in_force_is_forward_filled_between_settlements():

    frame = bars(6)
    aligned, _ = align_legs(frame, frame, funding_frame(np.array([0], dtype=np.int64), 0.0001, 4.0))
    assert np.allclose(funding_apr(aligned), 0.2190, rtol=1e-6)


def test_funding_with_no_stated_cadence_falls_back_to_eight_hours():
    """The REST path can lack it; the fallback must be explicit, not a silent zero."""

    times = np.arange(0, 3, dtype=np.int64) * HOUR
    frame = bars(3)
    funding = funding_frame(times, 0.0001, 8.0).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("interval_hours")
    )
    aligned, _ = align_legs(frame, frame, funding)
    assert funding_apr(aligned)[-1] == pytest.approx(0.1095, rel=1e-6)


def test_funding_intervals_reports_a_mid_month_change_instead_of_raising():
    """The guard that refused these months deleted the highest-funding episodes from every result."""

    csv = b"calc_time,funding_interval_hours,last_funding_rate\n1,8,0.0001\n2,4,0.0002\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("f.csv", csv)
    raw = buffer.getvalue()

    assert binance_archive.funding_intervals(raw, "uri") == [4.0, 8.0]
    with pytest.raises(binance_archive.ArchiveError, match="spec change"):
        binance_archive.funding_interval_hours(raw, "uri")


def test_parsed_funding_carries_the_interval_per_settlement():

    csv = b"calc_time,funding_interval_hours,last_funding_rate\n1,8,0.0001\n2,4,0.0002\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("f.csv", csv)
    parsed = binance_archive.parse_funding(buffer.getvalue(), "X", "uri")
    assert parsed["interval_hours"].to_list() == [8.0, 4.0]
