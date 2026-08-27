"""The trial registry (SPEC.md §10.4) and sealed-test-token issuance (§10.1).

Every `generate()` call with a distinct `(signal, params, symbol, period)` tuple inserts a row
**before** results are computed. `N` in the DSR formula reads from this table — never from a
constant, never from the length of a grid the caller happens to be holding.

Trials are counted **per symbol** (§8.1): 24 parameter combinations on a two-asset universe is
N=48, because searching a second asset is a second search.

The table is append-only and hash-chained: each row commits to the previous row's hash, so deleting
or editing a row is detectable by `verify_chain`. Deleting rows is a protocol violation; this class
offers no API to do it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptolab.validation.sealed import SealedPeriodError, TestPeriodToken

GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   INTEGER NOT NULL,
    signal       TEXT    NOT NULL,
    params_json  TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    period       TEXT    NOT NULL,
    strategy_family TEXT NOT NULL,
    note         TEXT    NOT NULL DEFAULT '',
    prev_hash    TEXT    NOT NULL,
    row_hash     TEXT    NOT NULL,
    UNIQUE (signal, params_json, symbol, period)
);
CREATE TABLE IF NOT EXISTS test_tokens (
    nonce           TEXT PRIMARY KEY,
    strategy_family TEXT NOT NULL,
    issued_at       INTEGER NOT NULL,
    consumed_at     INTEGER
);
CREATE TABLE IF NOT EXISTS registry_meta (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
"""


class RegistryError(RuntimeError):
    """Raised on a protocol violation against the registry."""


@dataclass(frozen=True, slots=True)
class Trial:
    """One registered trial."""

    id: int
    created_at: int
    signal: str
    params: dict[str, Any]
    symbol: str
    period: str
    strategy_family: str
    note: str
    prev_hash: str
    row_hash: str


class TrialRegistry:
    """Append-only, hash-chained SQLite trial registry."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._secret = self._load_or_create_secret()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TrialRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- secret ------------------------------------------------------------------

    def _load_or_create_secret(self) -> bytes:
        row = self._conn.execute(
            "SELECT value FROM registry_meta WHERE key = 'token_secret'"
        ).fetchone()
        if row is not None:
            return bytes(row["value"])
        secret = os.urandom(32)
        self._conn.execute(
            "INSERT INTO registry_meta (key, value) VALUES ('token_secret', ?)", (secret,)
        )
        self._conn.commit()
        return secret

    # ---- registration ------------------------------------------------------------

    def register(
        self,
        *,
        signal: str,
        params: dict[str, Any],
        symbol: str,
        period: str,
        strategy_family: str | None = None,
        note: str = "",
    ) -> Trial:
        """Register one trial before its results are computed. Idempotent on the natural key.

        Re-registering the same tuple returns the existing row rather than inflating `N`: repeating
        a computation is not a new statistical trial. Registering a *different* parameter tuple is,
        and it will lower every deflated Sharpe drawn from this registry.
        """
        params_json = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        family = strategy_family or signal
        existing = self._conn.execute(
            "SELECT * FROM trials WHERE signal=? AND params_json=? AND symbol=? AND period=?",
            (signal, params_json, symbol, period),
        ).fetchone()
        if existing is not None:
            return _row_to_trial(existing)

        created_at = _now_ms()
        prev_hash = self.head_hash()
        row_hash = _hash_row(
            prev_hash,
            created_at,
            signal=signal,
            params_json=params_json,
            symbol=symbol,
            period=period,
            strategy_family=family,
            note=note,
        )
        cursor = self._conn.execute(
            "INSERT INTO trials (created_at, signal, params_json, symbol, period, strategy_family,"
            " note, prev_hash, row_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (created_at, signal, params_json, symbol, period, family, note, prev_hash, row_hash),
        )
        self._conn.commit()
        return Trial(
            id=int(cursor.lastrowid or 0),
            created_at=created_at,
            signal=signal,
            params=params,
            symbol=symbol,
            period=period,
            strategy_family=family,
            note=note,
            prev_hash=prev_hash,
            row_hash=row_hash,
        )

    def register_grid(
        self,
        *,
        signal: str,
        grid: list[dict[str, Any]],
        symbols: list[str],
        period: str,
        strategy_family: str | None = None,
        note: str = "",
    ) -> list[Trial]:
        """Register a whole declared search: one trial per (params, symbol) pair — per §8.1."""
        return [
            self.register(
                signal=signal,
                params=params,
                symbol=symbol,
                period=period,
                strategy_family=strategy_family,
                note=note,
            )
            for symbol in symbols
            for params in grid
        ]

    # ---- counting ----------------------------------------------------------------

    def count(self, *, signal: str | None = None, strategy_family: str | None = None) -> int:
        """`N` for the DSR formula. Scope it to a family to deflate against that family's search."""
        clauses, args = [], []
        if signal is not None:
            clauses.append("signal = ?")
            args.append(signal)
        if strategy_family is not None:
            clauses.append("strategy_family = ?")
            args.append(strategy_family)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM trials {where}", args).fetchone()
        return int(row["n"])

    def all_trials(self) -> list[Trial]:
        rows = self._conn.execute("SELECT * FROM trials ORDER BY id").fetchall()
        return [_row_to_trial(r) for r in rows]

    def head_hash(self) -> str:
        row = self._conn.execute("SELECT row_hash FROM trials ORDER BY id DESC LIMIT 1").fetchone()
        return GENESIS_HASH if row is None else str(row["row_hash"])

    def verify_chain(self) -> bool:
        """Recompute the chain. False means a row was deleted or edited — a protocol violation."""
        prev = GENESIS_HASH
        for row in self._conn.execute("SELECT * FROM trials ORDER BY id").fetchall():
            if row["prev_hash"] != prev:
                return False
            expected = _hash_row(
                row["prev_hash"],
                int(row["created_at"]),
                signal=row["signal"],
                params_json=row["params_json"],
                symbol=row["symbol"],
                period=row["period"],
                strategy_family=row["strategy_family"],
                note=row["note"],
            )
            if expected != row["row_hash"]:
                return False
            prev = str(row["row_hash"])
        return True

    # ---- sealed test period ------------------------------------------------------

    def issue_test_token(self, strategy_family: str) -> TestPeriodToken:
        """Issue the one-and-only sealed-test token for `strategy_family` (§10.1).

        A second issuance for the same family raises. This is the mechanism that makes the test
        period a one-touch resource; there is no override.
        """
        prior = self._conn.execute(
            "SELECT nonce, consumed_at FROM test_tokens WHERE strategy_family = ?",
            (strategy_family,),
        ).fetchone()
        if prior is not None:
            state = "consumed" if prior["consumed_at"] is not None else "issued but unspent"
            raise SealedPeriodError(
                f"the sealed test period has already been opened for strategy family "
                f"{strategy_family!r} (token {state}). It is a one-touch resource (SPEC.md §10.1); "
                "a second read is not available at any privilege level."
            )
        token = TestPeriodToken.mint(strategy_family, self._secret)
        self._conn.execute(
            "INSERT INTO test_tokens (nonce, strategy_family, issued_at) VALUES (?,?,?)",
            (token.nonce, strategy_family, _now_ms()),
        )
        self._conn.commit()
        return token

    def consume_test_token(self, token: TestPeriodToken) -> bool:
        """Verify and spend a token. Returns False if invalid or already spent."""
        if not token.verify(self._secret):
            return False
        row = self._conn.execute(
            "SELECT strategy_family, consumed_at FROM test_tokens WHERE nonce = ?", (token.nonce,)
        ).fetchone()
        if row is None or row["strategy_family"] != token.strategy_family:
            return False
        if row["consumed_at"] is not None:
            return False
        self._conn.execute(
            "UPDATE test_tokens SET consumed_at = ? WHERE nonce = ?", (_now_ms(), token.nonce)
        )
        self._conn.commit()
        return True

    def bind_store(self, store: Any) -> Any:
        """Attach this registry's verifier to a `ParquetStore` so it can check sealed-period tokens."""
        store._verify = self.consume_test_token
        return store


def _row_to_trial(row: sqlite3.Row) -> Trial:
    return Trial(
        id=int(row["id"]),
        created_at=int(row["created_at"]),
        signal=str(row["signal"]),
        params=json.loads(row["params_json"]),
        symbol=str(row["symbol"]),
        period=str(row["period"]),
        strategy_family=str(row["strategy_family"]),
        note=str(row["note"]),
        prev_hash=str(row["prev_hash"]),
        row_hash=str(row["row_hash"]),
    )


def _hash_row(
    prev_hash: str,
    created_at: int,
    *,
    signal: str,
    params_json: str,
    symbol: str,
    period: str,
    strategy_family: str,
    note: str,
) -> str:
    payload = "|".join(
        [prev_hash, str(created_at), signal, params_json, symbol, period, strategy_family, note]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _now_ms() -> int:
    return int(dt.datetime.now(tz=dt.UTC).timestamp() * 1000)
