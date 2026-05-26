"""Lightweight SQLite migration runner.

Esther's per-store schemas have lived as inline ``CREATE TABLE IF NOT
EXISTS`` strings since the prototype days. That's been fine for one
table per store, but the "real" path forward is a versioned migration
trail so a column add doesn't require manual SQL against existing
databases.

This module is the smallest viable migrator that fits the existing
lazy-init / threading-lock pattern:

* One ``_schema_migrations`` table per database file (auto-created on
  first ``run_migrations`` call).
* A migration is a ``(version: int, name: str, apply_fn: Callable)``
  triple. ``version`` MUST be strictly increasing within a database;
  ``name`` is a free-form label recorded for forensics.
* ``run_migrations(conn, migrations)`` is idempotent — already-applied
  versions are skipped, missing versions are applied in order.
* Atomicity: each migration's ``apply`` callable is responsible for
  its own transaction discipline (SQLite's autocommit-by-default mode
  + the fact that ``executescript`` issues an implicit ``COMMIT`` make
  external wrapping unreliable). The version row is inserted only
  after ``apply`` returns cleanly, so a mid-migration crash leaves the
  database short of one version, and re-running re-applies that
  migration. Keep migrations small and additive (``CREATE TABLE IF
  NOT EXISTS`` / ``ALTER TABLE ADD COLUMN``) so the re-try is a
  no-op when the partial work landed.

Why not Alembic
---------------
Alembic is the right tool for a Postgres-backed multi-developer
codebase with branching migration histories. At our scale (single
process, one SQLite file per store, additive-only schema changes), the
table-per-store + integer-version model is sufficient and adds zero
dependencies. The dependency injection is the seam for swapping to
Alembic when we move off SQLite — every store calls
``run_migrations`` once at first connect; replacing that with an
``alembic upgrade head`` is a small surgical change.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Migration:
    """One ordered schema-evolution step."""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


_SCHEMA_TABLE = "_schema_migrations"

# Composite primary key on (namespace, version). Two stores can share
# a SQLite file (the production layout puts UserStore + WatchlistStore
# both at users.db) and each owns its own independent version trail.
_CREATE_SCHEMA_TABLE = f"""
CREATE TABLE IF NOT EXISTS {_SCHEMA_TABLE} (
    namespace TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    PRIMARY KEY (namespace, version)
);
"""


def run_migrations(
    conn: sqlite3.Connection,
    migrations: list[Migration],
    *,
    namespace: str,
) -> list[int]:
    """Apply any not-yet-applied migrations to ``conn`` under ``namespace``.

    Returns the list of newly-applied versions (in order). An empty
    list means everything was already at head. Idempotent — call once
    per store connection without checking elsewhere.

    Args:
        conn: An open SQLite connection. Caller owns lifetime + locking.
        migrations: Strictly version-increasing list. Out-of-order
            versions raise immediately so a bad rebase fails loudly at
            startup instead of silently re-running an old migration.
        namespace: Per-store label. Distinct stores sharing a SQLite
            file (e.g. ``UserStore`` and ``WatchlistStore`` at
            ``users.db``) each pass their own namespace so version
            trails don't collide on the shared ``_schema_migrations``
            table. Convention: the store class name in snake_case.
    """
    if not namespace:
        raise ValueError("namespace must be a non-empty string")
    _validate_ordering(migrations)
    conn.execute(_CREATE_SCHEMA_TABLE)
    applied = _applied_versions(conn, namespace)
    newly_applied: list[int] = []
    for migration in migrations:
        if migration.version in applied:
            continue
        # Run the migration's DDL — the callable is responsible for its
        # own transaction shape if it needs one. We then record the
        # version row only on success: a thrown apply leaves the
        # ``_schema_migrations`` table unchanged so the next run
        # re-tries this migration. Migrations should be idempotent
        # (``CREATE TABLE IF NOT EXISTS`` / ``ALTER TABLE`` guarded by
        # PRAGMA checks) to keep the re-try safe.
        migration.apply(conn)
        conn.execute(
            f"INSERT INTO {_SCHEMA_TABLE} "
            "(namespace, version, name, applied_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (namespace, migration.version, migration.name),
        )
        log.info(
            "migrations.applied",
            namespace=namespace,
            version=migration.version,
            name=migration.name,
        )
        newly_applied.append(migration.version)
    return newly_applied


def _applied_versions(
    conn: sqlite3.Connection, namespace: str
) -> frozenset[int]:
    cur = conn.execute(
        f"SELECT version FROM {_SCHEMA_TABLE} WHERE namespace = ?",
        (namespace,),
    )
    return frozenset(int(row[0]) for row in cur.fetchall())


def _validate_ordering(migrations: list[Migration]) -> None:
    if not migrations:
        return
    seen: set[int] = set()
    last = -1
    for m in migrations:
        if m.version in seen:
            raise ValueError(f"duplicate migration version {m.version}")
        if m.version <= last:
            raise ValueError(
                f"migration versions must be strictly increasing: "
                f"version {m.version} ({m.name!r}) follows {last}"
            )
        seen.add(m.version)
        last = m.version


__all__ = ["Migration", "run_migrations"]
