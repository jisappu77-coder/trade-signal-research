"""The live readout. Mocked transport — the test suite never touches the network."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from cryptolab.data.sources.okx_live import LiveDataError, LiveQuote, fetch_all, fetch_quote
from cryptolab.live import (
    LiveSignal,
    append_observation,
    evaluate,
    read_observations,
    summarise_watch,
)

NOW = 1_788_468_302_845
NEXT = NOW + 8 * 3_600_000


def okx_handler(funding: str = "0.0003", spot: str = "80000", perp: str = "80100"):
    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        inst = request.url.params.get("instId", "")
        if "funding-rate" in path:
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "data": [
                        {
                            "instId": inst,
                            "fundingRate": funding,
                            "fundingTime": str(NOW),
                            "nextFundingTime": str(NEXT),
                        }
                    ],
                },
            )
        price = perp if "SWAP" in inst else spot
        return httpx.Response(
            200, json={"code": "0", "data": [{"instId": inst, "last": price, "ts": str(NOW)}]}
        )

    return handle


def quote(**overrides) -> LiveQuote:
    base = {
        "symbol": "BTCUSDT",
        "observed_at": NOW,
        "spot_price": 80_000.0,
        "perp_price": 80_100.0,
        "funding_rate": 0.0003,
        "funding_interval_hours": 8.0,
        "next_funding_time": NEXT,
    }
    return LiveQuote(**{**base, **overrides})


# ---- the venue adapter --------------------------------------------------------------


def test_fetch_reads_both_legs_and_the_funding_rate():
    client = httpx.AsyncClient(transport=httpx.MockTransport(okx_handler()))
    got = asyncio.run(fetch_quote(client, "BTCUSDT"))
    assert got.spot_price == 80_000.0
    assert got.perp_price == 80_100.0
    assert got.funding_rate == pytest.approx(0.0003)


def test_the_funding_interval_comes_from_the_venue():
    """§5.1: never assume 8h. OKX reports the next settlement, so the gap is measured."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(okx_handler()))
    assert asyncio.run(fetch_quote(client, "BTCUSDT")).funding_interval_hours == pytest.approx(8.0)


def test_an_unknown_symbol_is_refused():
    client = httpx.AsyncClient(transport=httpx.MockTransport(okx_handler()))
    with pytest.raises(LiveDataError, match="unknown symbol"):
        asyncio.run(fetch_quote(client, "DOGEUSDT"))


def test_a_venue_error_is_surfaced_not_swallowed():
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "50011", "msg": "rate limited", "data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    with pytest.raises(LiveDataError, match="rate limited"):
        asyncio.run(fetch_quote(client, "BTCUSDT"))


def test_fetch_all_reads_every_symbol():
    client = httpx.AsyncClient(transport=httpx.MockTransport(okx_handler()))
    assert len(asyncio.run(fetch_all(client, ["BTCUSDT", "ETHUSDT"]))) == 2


def test_basis_is_perp_against_spot():
    assert quote(spot_price=80_000.0, perp_price=80_080.0).basis_bps == pytest.approx(10.0)


def test_funding_apr_annualises_from_the_reported_interval():
    assert quote(funding_rate=0.0001, funding_interval_hours=8.0).funding_apr == pytest.approx(0.1095)
    assert quote(funding_rate=0.0001, funding_interval_hours=4.0).funding_apr == pytest.approx(0.219)


# ---- the signal ---------------------------------------------------------------------


def test_low_funding_does_not_fire():
    signal = evaluate(quote(funding_rate=0.00003), holding_days=7.0)
    assert not signal.fires
    assert signal.headroom_apr < 0
    assert "STAND DOWN" in signal.action


def test_high_funding_fires():
    signal = evaluate(quote(funding_rate=0.005), holding_days=7.0)
    assert signal.fires
    assert signal.headroom_apr > 0
    assert "DEPLOY" in signal.action


def test_negative_funding_never_fires():
    """A short perp pays when funding is negative — the opposite of the carry trade."""
    assert not evaluate(quote(funding_rate=-0.003), holding_days=7.0).fires


def test_a_shorter_hold_demands_more_funding():
    """The same rate can fire on a weekly hold and not on a daily one."""
    weekly = evaluate(quote(funding_rate=0.0006), holding_days=7.0)
    daily = evaluate(quote(funding_rate=0.0006), holding_days=1.0)
    assert daily.entry_threshold_apr > weekly.entry_threshold_apr


def test_a_worse_cost_regime_raises_the_bar():
    cheap = evaluate(quote(), regime_name="optimistic")
    dear = evaluate(quote(), regime_name="stressed")
    assert dear.entry_threshold_apr > cheap.entry_threshold_apr


def test_the_readout_names_the_venue():
    """The backtest is Binance and this is OKX; the reading must not hide which."""
    assert evaluate(quote()).venue == "okx"


def test_the_line_is_readable():
    line = evaluate(quote(funding_rate=0.005)).line()
    assert "SIGNAL" in line and "BTCUSDT" in line and "APR" in line


# ---- the watch log ------------------------------------------------------------------


def test_observations_round_trip(tmp_path):
    path = tmp_path / "watch.jsonl"
    append_observation(evaluate(quote()), path)
    append_observation(evaluate(quote(symbol="ETHUSDT")), path)
    rows = read_observations(path)
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"BTCUSDT", "ETHUSDT"}


def test_reading_a_missing_log_is_empty_not_an_error(tmp_path):
    assert read_observations(tmp_path / "nothing.jsonl") == []


def test_summary_reports_how_often_the_signal_fired(tmp_path):
    path = tmp_path / "watch.jsonl"
    for rate in (0.00003, 0.005, 0.00003):
        append_observation(evaluate(quote(funding_rate=rate)), path)
    summary = summarise_watch(read_observations(path))
    assert "3 observations" in summary
    assert "1 fired" in summary


def test_summary_of_nothing_says_so():
    assert summarise_watch([]) == "no observations recorded"


def test_summary_reports_the_closest_approach(tmp_path):
    """When nothing fires, how close it came is the informative number."""
    path = tmp_path / "watch.jsonl"
    for rate in (0.00001, 0.0002):
        append_observation(evaluate(quote(funding_rate=rate)), path)
    assert "closest approach" in summarise_watch(read_observations(path))


def test_the_signal_serialises_for_the_log():
    payload = evaluate(quote()).to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert "action" in payload
    assert isinstance(payload["fires"], bool)


def test_evaluate_returns_a_frozen_record():
    signal = evaluate(quote())
    assert isinstance(signal, LiveSignal)
    with pytest.raises(AttributeError):
        signal.fires = True  # type: ignore[misc]
