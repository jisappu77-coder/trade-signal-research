from __future__ import annotations

import sqlite3

import pytest

from cryptolab.validation.registry import TrialRegistry


@pytest.fixture
def registry(tmp_path):
    with TrialRegistry(tmp_path / "registry.sqlite") as r:
        yield r


def test_registering_a_trial_increments_n(registry):
    assert registry.count() == 0
    registry.register(signal="tsmom", params={"L": 96}, symbol="BTCUSDT", period="train")
    assert registry.count() == 1


def test_reregistering_the_same_tuple_does_not_inflate_n(registry):
    for _ in range(5):
        registry.register(signal="tsmom", params={"L": 96}, symbol="BTCUSDT", period="train")
    assert registry.count() == 1


def test_trials_are_counted_per_symbol_not_per_universe(registry):
    """§8.1: 24 combinations on a two-asset universe is N=48."""
    grid = [{"L": lookback, "H": halflife} for lookback in (24, 48, 96, 168) for halflife in (36, 72, 144)]
    assert len(grid) == 12
    registry.register_grid(signal="tsmom", grid=grid, symbols=["BTCUSDT", "ETHUSDT"], period="train")
    assert registry.count() == 24

    # The full §8.1 space also varies the bar, giving 24 combinations and N=48.
    full = [dict(p, bar=bar) for p in grid for bar in ("1h", "4h")]
    registry.register_grid(signal="tsmom_full", grid=full, symbols=["BTCUSDT", "ETHUSDT"], period="train")
    assert registry.count(signal="tsmom_full") == 48


def test_count_can_be_scoped_to_a_family(registry):
    registry.register(signal="tsmom", params={"L": 24}, symbol="BTCUSDT", period="train")
    registry.register(signal="carry", params={"t": 1}, symbol="BTCUSDT", period="train")
    assert registry.count(signal="tsmom") == 1
    assert registry.count() == 2


def test_params_survive_a_round_trip(registry):
    registry.register(signal="tsmom", params={"L": 96, "sigma_target": 0.4}, symbol="BTCUSDT", period="train")
    assert registry.all_trials()[0].params == {"L": 96, "sigma_target": 0.4}


def test_hash_chain_is_intact_after_normal_use(registry):
    registry.register_grid(
        signal="tsmom", grid=[{"L": i} for i in range(10)], symbols=["BTCUSDT"], period="train"
    )
    assert registry.verify_chain()


def test_deleting_a_row_is_detectable(tmp_path):
    """Deleting rows is a protocol violation; the chain makes it visible."""
    path = tmp_path / "registry.sqlite"
    registry = TrialRegistry(path)
    registry.register_grid(
        signal="tsmom", grid=[{"L": i} for i in range(5)], symbols=["BTCUSDT"], period="train"
    )
    assert registry.verify_chain()
    registry.close()

    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM trials WHERE id = 3")
    conn.commit()
    conn.close()

    reopened = TrialRegistry(path)
    assert not reopened.verify_chain()
    reopened.close()


def test_editing_a_row_is_detectable(tmp_path):
    path = tmp_path / "registry.sqlite"
    registry = TrialRegistry(path)
    registry.register(signal="tsmom", params={"L": 96}, symbol="BTCUSDT", period="train")
    registry.close()

    conn = sqlite3.connect(path)
    conn.execute("UPDATE trials SET params_json = '{\"L\": 24}' WHERE id = 1")
    conn.commit()
    conn.close()

    reopened = TrialRegistry(path)
    assert not reopened.verify_chain()
    reopened.close()


def test_registry_persists_across_sessions(tmp_path):
    path = tmp_path / "registry.sqlite"
    first = TrialRegistry(path)
    first.register(signal="tsmom", params={"L": 96}, symbol="BTCUSDT", period="train")
    first.close()

    second = TrialRegistry(path)
    assert second.count() == 1 and second.verify_chain()
    second.close()


def test_n_reflects_an_expanded_search(registry):
    """Widening a grid is registrable and raises N — the honest cost of a second look."""
    registry.register_grid(
        signal="tsmom", grid=[{"L": i} for i in (24, 48)], symbols=["BTCUSDT"], period="train"
    )
    before = registry.count()
    registry.register_grid(
        signal="tsmom", grid=[{"L": i} for i in (24, 48, 96, 168)], symbols=["BTCUSDT"], period="train"
    )
    assert registry.count() == before + 2
