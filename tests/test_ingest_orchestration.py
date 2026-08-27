"""Ingestion and CLI tests against a mocked transport — no network access in the test suite."""

from __future__ import annotations

import asyncio
import io
import zipfile

import httpx
import pytest
from typer.testing import CliRunner

from cryptolab.cli import app
from cryptolab.data.ingest import (
    collect_open_interest,
    ingest_funding,
    ingest_funding_archive,
    ingest_klines,
)
from cryptolab.data.sources import binance_api

runner = CliRunner()

ROWS = "\n".join(
    f"{1_546_300_800_000 + i * 3_600_000},100.0,101.0,99.0,100.5,10.0,"
    f"{1_546_300_800_000 + (i + 1) * 3_600_000 - 1},1000.0,5,5.0,500.0,0"
    for i in range(24)
)


def _kline_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("k.csv", ROWS)
    return buffer.getvalue()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_ingest_klines_writes_and_gates_each_month(store):
    def handler(request: httpx.Request) -> httpx.Response:
        if "2019-01" in str(request.url):
            return httpx.Response(200, content=_kline_zip())
        return httpx.Response(404)

    results = asyncio.run(
        ingest_klines(
            store, "BTCUSDT", "1h", "2019-01-01", "2019-03-01", client=_client(handler)
        )
    )
    assert len(results) == 3
    assert results[0].rows == 24 and results[0].report is not None
    assert [r.skipped for r in results[1:]] == ["404", "404"]

    stored = store.read(
        "ohlcv", exchange="binance", symbol="BTCUSDT", start="2019-01-01", end="2019-02-01"
    )
    assert stored.height == 24


def test_a_missing_month_is_skipped_not_fatal(store):
    results = asyncio.run(
        ingest_klines(
            store, "BTCUSDT", "1h", "2019-01-01", "2019-01-31",
            client=_client(lambda _: httpx.Response(404)),
        )
    )
    assert results[0].skipped == "404" and results[0].rows == 0


def test_a_server_error_propagates(store):
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            ingest_klines(
                store, "BTCUSDT", "1h", "2019-01-01", "2019-01-31",
                client=_client(lambda _: httpx.Response(500)),
            )
        )


def test_ingest_funding_stops_on_a_short_page(store):
    """A page shorter than the limit means the history is exhausted — stop, do not re-poll."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(int(request.url.params["startTime"]))
        return httpx.Response(
            200,
            json=[
                {
                    "fundingTime": 1_546_300_800_000 + i * 28_800_000,
                    "fundingRate": "0.0001",
                    "markPrice": "20000",
                }
                for i in range(3)
            ],
        )

    result = asyncio.run(
        ingest_funding(store, "BTCUSDT", "2019-01-01", "2019-01-05", client=_client(handler))
    )
    assert result.rows == 3
    assert len(calls) == 1


def test_ingest_funding_paginates_forward_through_full_pages(store):
    """A full page means more history remains; the cursor advances past the last settlement."""
    starts: list[int] = []
    page_size = binance_api.FUNDING_PAGE_LIMIT

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startTime"])
        starts.append(start)
        if len(starts) > 2:
            return httpx.Response(200, json=[])
        base = 1_546_300_800_000 + (len(starts) - 1) * page_size * 28_800_000
        return httpx.Response(
            200,
            json=[
                {
                    "fundingTime": base + i * 28_800_000,
                    "fundingRate": "0.0001",
                    "markPrice": "20000",
                }
                for i in range(page_size)
            ],
        )

    result = asyncio.run(
        ingest_funding(store, "BTCUSDT", "2019-01-01", "2030-01-01", client=_client(handler))
    )
    assert result.rows == 2 * page_size
    assert starts == sorted(starts) and len(starts) == 3


def test_ingest_funding_gates_on_the_cap_breach(store):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"fundingTime": 1_546_300_800_000, "fundingRate": "0.05", "markPrice": "20000"}],
        )

    result = asyncio.run(
        ingest_funding(store, "BTCUSDT", "2019-01-01", "2019-01-02", client=_client(handler))
    )
    assert result.report is not None and not result.report.passed


def test_collect_open_interest_writes_each_symbol(store):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "timestamp": 1_546_300_800_000,
                    "sumOpenInterest": "100",
                    "sumOpenInterestValue": "2000000",
                }
            ],
        )

    results = asyncio.run(
        collect_open_interest(store, ["BTCUSDT", "ETHUSDT"], client=_client(handler))
    )
    assert [r.symbol for r in results] == ["BTCUSDT", "ETHUSDT"]
    assert all(r.rows == 1 for r in results)


def test_fetch_open_interest_respects_the_page_limit():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["limit"])
        return httpx.Response(200, json=[])

    asyncio.run(binance_api.fetch_open_interest(_client(handler), "BTCUSDT", limit=5000))
    assert seen == [str(binance_api.OI_PAGE_LIMIT)]


# ---- CLI ---------------------------------------------------------------------------


def test_init_prints_the_sealed_period_and_the_oi_warning():
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "sealed test period" in result.stdout
    assert "PERMANENTLY UNRECOVERABLE" in result.stdout


def test_harness_check_exercises_the_validation_stack():
    result = runner.invoke(app, ["harness-check"])
    assert result.exit_code == 0
    for line in ("DSR on pure noise", "PBO on pure noise", "fold dispersion"):
        assert line in result.stdout


def test_registry_status_reports_n_and_chain_health(tmp_path, monkeypatch):
    config = tmp_path / "base.yaml"
    config.write_text(
        "universe: [BTCUSDT]\nexchange: binance\nbars: [1h]\n"
        f"data_root: {tmp_path / 'data'}\nregistry_path: {tmp_path / 'r.sqlite'}\n"
        f"report_root: {tmp_path / 'reports'}\n"
        "splits:\n"
        "  train: {start: '2019-01-01', end: '2022-12-31'}\n"
        "  validation: {start: '2023-01-01', end: '2024-06-30'}\n"
        "  test: {start: '2024-07-01', end: null}\n"
    )
    result = runner.invoke(app, ["registry-status", "--config", str(config)])
    assert result.exit_code == 0
    assert "trials N" in result.stdout and "intact" in result.stdout


def test_there_is_no_run_strategy_command():
    """Phase 4 does not exist yet, and the CLI must not pretend otherwise."""
    assert "run-strategy" not in runner.invoke(app, ["--help"]).stdout


def _cli_config(tmp_path) -> str:
    config = tmp_path / "base.yaml"
    config.write_text(
        "universe: [BTCUSDT, ETHUSDT]\nexchange: binance\nbars: [1h]\n"
        f"data_root: {tmp_path / 'data'}\nregistry_path: {tmp_path / 'r.sqlite'}\n"
        f"report_root: {tmp_path / 'reports'}\n"
        "splits:\n"
        "  train: {start: '2019-01-01', end: '2022-12-31'}\n"
        "  validation: {start: '2023-01-01', end: '2024-06-30'}\n"
        "  test: {start: '2024-07-01', end: null}\n"
    )
    return str(config)


def _patch_transport(monkeypatch, handler) -> None:
    """Force every AsyncClient the CLI builds onto a mock transport."""
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


def test_cli_ingest_reports_bar_counts(tmp_path, monkeypatch):
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_kline_zip()))
    result = runner.invoke(
        app,
        ["ingest", "BTCUSDT", "--start", "2019-01-01", "--end", "2019-01-31",
         "--config", _cli_config(tmp_path)],
    )
    assert result.exit_code == 0
    assert "24 bars" in result.stdout


def test_cli_ingest_exits_nonzero_on_a_quality_failure(tmp_path, monkeypatch):
    """A month that fails the §6 gate must fail the command, not warn and continue."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        # A single bar, then a 10-hour hole: missing bars fail the gate.
        zf.writestr(
            "k.csv",
            "1546300800000,100,101,99,100.5,10,1546304399999,1000,5,5,500,0\n"
            "1546336800000,100,101,99,100.5,10,1546340399999,1000,5,5,500,0",
        )
    holed = buffer.getvalue()
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=holed))

    result = runner.invoke(
        app,
        ["ingest", "BTCUSDT", "--start", "2019-01-01", "--end", "2019-01-31",
         "--config", _cli_config(tmp_path)],
    )
    assert result.exit_code == 1
    assert "QUALITY FAIL" in result.stdout


def test_cli_ingest_funding(tmp_path, monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json=[{"fundingTime": 1_546_300_800_000, "fundingRate": "0.0001", "markPrice": "20000"}],
        ),
    )
    result = runner.invoke(
        app,
        ["ingest-funding", "BTCUSDT", "--source", "rest",
         "--start", "2019-01-01", "--end", "2019-01-05",
         "--config", _cli_config(tmp_path)],
    )
    assert result.exit_code == 0 and "1 settlements" in result.stdout


def test_cli_ingest_funding_fails_on_a_cap_breach(tmp_path, monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json=[{"fundingTime": 1_546_300_800_000, "fundingRate": "0.05", "markPrice": "20000"}],
        ),
    )
    result = runner.invoke(
        app,
        ["ingest-funding", "BTCUSDT", "--source", "rest",
         "--start", "2019-01-01", "--end", "2019-01-05",
         "--config", _cli_config(tmp_path)],
    )
    assert result.exit_code == 1 and "funding_cap_breach" in result.stdout


def test_cli_collect_oi_warns_before_collecting(tmp_path, monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json=[
                {
                    "timestamp": 1_546_300_800_000,
                    "sumOpenInterest": "100",
                    "sumOpenInterestValue": "2000000",
                }
            ],
        ),
    )
    result = runner.invoke(app, ["collect-oi", "--config", _cli_config(tmp_path)])
    assert result.exit_code == 0
    assert "PERMANENTLY UNRECOVERABLE" in result.stdout
    assert "BTCUSDT: 1 open-interest points" in result.stdout


def test_ingest_funding_archive_walks_months(store):
    def handler(request: httpx.Request) -> httpx.Response:
        if "fundingRate" not in str(request.url):
            return httpx.Response(404)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(
                "f.csv",
                "calc_time,funding_interval_hours,last_funding_rate\n"
                + "\n".join(
                    f"{1_577_836_800_000 + i * 28_800_000},8,0.0001" for i in range(3)
                ),
            )
        return httpx.Response(200, content=buffer.getvalue())

    results = asyncio.run(
        ingest_funding_archive(
            store, "BTCUSDT", "2020-01-01", "2020-03-31", client=_client(handler)
        )
    )
    assert len(results) == 3
    assert all(r.interval == "8h" for r in results)
    assert sum(r.rows for r in results) == 9


def test_ingest_funding_archive_skips_a_missing_month(store):
    results = asyncio.run(
        ingest_funding_archive(
            store, "BTCUSDT", "2020-01-01", "2020-01-31",
            client=_client(lambda _: httpx.Response(404)),
        )
    )
    assert results[0].skipped == "404"


def test_cli_ingest_funding_rejects_an_unknown_source(tmp_path):
    result = runner.invoke(
        app,
        ["ingest-funding", "BTCUSDT", "--source", "carrier-pigeon",
         "--config", _cli_config(tmp_path)],
    )
    assert result.exit_code == 2 and "unknown source" in result.stdout
