"""Per-user watchlist persistence.

SQLite-backed, lazy-init, mirrors the :class:`UserStore` /
:class:`HealthStore` conventions. One table:

* ``watchlists`` — ``(user_id, symbol, added_at, sort_order)`` with a
  unique pair index. Foreign-keys into ``users`` so deleting an account
  cascades the user's watchlist away.

Sharing the file with UserStore
-------------------------------
The default deployment puts UserStore and WatchlistStore at the same
``users.db`` path. SQLite WAL mode supports multiple connections to
the same file from one process; the per-store ``threading.Lock``
serializes writes inside each store, and WAL handles the cross-store
ordering. Tests can pass the same ``tmp_path / "users.db"`` to both.

Why a separate store class
--------------------------
UserStore's job is "accounts + sessions." Adding watchlist methods
to it would broaden a security-sensitive class. Keeping watchlist in
its own module lets the auth code stay tightly scoped while sharing
the underlying file for backup/access purposes.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.data.migrations import Migration, run_migrations
from src.utils.logging import get_logger

log = get_logger(__name__)


# No FK to users(id). The schemas of UserStore and WatchlistStore
# share a SQLite file but materialize their tables independently —
# whichever store gets its first call wins the table-creation race.
# An FK on a non-yet-existing parent table errors at INSERT time
# under PRAGMA foreign_keys=ON, so we keep the cross-table integrity
# at the application layer (the future delete-account flow will
# explicitly purge watchlists for the user before removing the row).
# v1 — initial watchlists table. Same idempotent shape as before,
# wrapped in a Migration so future column / index additions follow the
# versioned trail in :mod:`src.data.migrations`.
_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    added_at TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_watchlists_user
    ON watchlists(user_id, sort_order, id);
"""


def _apply_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_V1_SCHEMA)


_MIGRATIONS = [
    Migration(version=1, name="watchlists_init", apply=_apply_v1),
]
_MIGRATION_NAMESPACE = "watchlist_store"


@dataclass(frozen=True)
class WatchlistEntry:
    """One symbol on a user's watchlist."""

    symbol: str
    added_at: datetime
    sort_order: int

    def to_wire(self) -> dict[str, str | int]:
        """JSON-safe shape for ``/api/watchlist`` responses."""
        return {
            "symbol": self.symbol,
            "added_at": self.added_at.isoformat(),
            "sort_order": self.sort_order,
        }


class SymbolAlreadyWatched(ValueError):
    """Raised when ``add`` is called with a symbol already in the list."""


class WatchlistStore:
    """SQLite-backed per-user watchlist.

    Same lazy-init contract as :class:`UserStore`: construction is
    free; the DB file + schema materialize on the first read/write.
    Per-store ``threading.Lock`` serializes writes; WAL covers the
    cross-store path when sharing a file with UserStore.
    """

    def __init__(
        self, db_path: Path, *, check_same_thread: bool = False
    ) -> None:
        self._db_path = db_path
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=self._check_same_thread,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Versioned migrations — see src/data/migrations.py. v1 is the
        # historical CREATE TABLE IF NOT EXISTS block; future changes
        # append a v2 Migration here. Namespace separates this store's
        # version trail from UserStore when both share ``users.db``.
        run_migrations(conn, _MIGRATIONS, namespace=_MIGRATION_NAMESPACE)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----- queries ----------------------------------------------------

    def list_for(self, user_id: int) -> list[WatchlistEntry]:
        """Return the user's watchlist in display order.

        Ordered by ``sort_order`` then insert id so a freshly seeded
        list comes out in seed order, and later additions append.
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT symbol, added_at, sort_order
            FROM watchlists
            WHERE user_id = ?
            ORDER BY sort_order ASC, id ASC
            """,
            (user_id,),
        )
        return [
            WatchlistEntry(
                symbol=row[0],
                added_at=_parse_iso(row[1]),
                sort_order=int(row[2]),
            )
            for row in cur.fetchall()
        ]

    def contains(self, user_id: int, symbol: str) -> bool:
        """Cheap membership check. Lower-cases the comparison."""
        sym = _normalize_symbol(symbol)
        if not sym:
            return False
        conn = self._connect()
        cur = conn.execute(
            "SELECT 1 FROM watchlists WHERE user_id = ? AND symbol = ?",
            (user_id, sym),
        )
        return cur.fetchone() is not None

    # ----- mutations --------------------------------------------------

    def add(self, user_id: int, symbol: str) -> WatchlistEntry:
        """Append ``symbol`` to the user's watchlist.

        Symbol is upper-cased + trimmed. Raises
        :class:`SymbolAlreadyWatched` on duplicate (the UNIQUE pair
        index prevents the insert; we translate the IntegrityError
        rather than masking it).
        """
        sym = _normalize_symbol(symbol)
        if not sym:
            raise ValueError("symbol must not be empty")
        added_at = datetime.now(UTC)
        with self._lock:
            conn = self._connect()
            # ``sort_order`` = current max + 1 so new symbols append.
            # Same transaction as the insert: the SELECT result is
            # only used to compute the next slot, so a concurrent
            # add from the same user could race to the same sort_order
            # — that's fine, it's just display order.
            cur = conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 "
                "FROM watchlists WHERE user_id = ?",
                (user_id,),
            )
            next_order = int(cur.fetchone()[0])
            try:
                conn.execute(
                    """
                    INSERT INTO watchlists
                        (user_id, symbol, added_at, sort_order)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, sym, added_at.isoformat(), next_order),
                )
            except sqlite3.IntegrityError as exc:
                if "watchlists" in str(exc).lower():
                    raise SymbolAlreadyWatched(sym) from None
                raise
        return WatchlistEntry(
            symbol=sym, added_at=added_at, sort_order=next_order
        )

    def remove(self, user_id: int, symbol: str) -> bool:
        """Drop ``symbol`` from the user's watchlist. Returns True iff
        a row was removed (False for not-present / empty input)."""
        sym = _normalize_symbol(symbol)
        if not sym:
            return False
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM watchlists WHERE user_id = ? AND symbol = ?",
                (user_id, sym),
            )
            return cur.rowcount > 0

    def delete_all_for(self, user_id: int) -> int:
        """Drop every watchlist row for ``user_id``. Returns count.

        Called from the account-delete cross-store purge hook in
        ``app.py``. WatchlistStore deliberately has no FK back to
        ``users(id)`` (the two stores' tables materialize
        independently and an FK on a not-yet-existing parent table
        races at INSERT), so the user-row CASCADE doesn't wipe
        watchlists — this method does.
        """
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM watchlists WHERE user_id = ?", (user_id,)
            )
            return int(cur.rowcount or 0)

    def seed_default(
        self, user_id: int, symbols: Iterable[str]
    ) -> list[WatchlistEntry]:
        """Bulk-insert ``symbols`` for a new user.

        Idempotent on duplicates — silently skips symbols already
        present. Called from the signup flow so first-run users land
        on a populated dashboard instead of an empty list.
        """
        added_at = datetime.now(UTC)
        seeded: list[WatchlistEntry] = []
        with self._lock:
            conn = self._connect()
            # Seed in the order given. Find current max so we don't
            # collide with any rows the user may have already added
            # (shouldn't happen on signup but seed is also reusable).
            cur = conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 "
                "FROM watchlists WHERE user_id = ?",
                (user_id,),
            )
            next_order = int(cur.fetchone()[0])
            for raw in symbols:
                sym = _normalize_symbol(raw)
                if not sym:
                    continue
                try:
                    conn.execute(
                        """
                        INSERT INTO watchlists
                            (user_id, symbol, added_at, sort_order)
                        VALUES (?, ?, ?, ?)
                        """,
                        (user_id, sym, added_at.isoformat(), next_order),
                    )
                except sqlite3.IntegrityError:
                    # Already present — skip rather than abort the seed.
                    continue
                seeded.append(
                    WatchlistEntry(
                        symbol=sym,
                        added_at=added_at,
                        sort_order=next_order,
                    )
                )
                next_order += 1
        return seeded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_symbol(symbol: str) -> str:
    """Trim + upper-case. Tickers in our universe are uppercase."""
    return (symbol or "").strip().upper()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "SymbolAlreadyWatched",
    "WatchlistEntry",
    "WatchlistStore",
]
