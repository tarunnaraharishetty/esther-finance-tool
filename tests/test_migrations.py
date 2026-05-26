"""Tests for the SQLite migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.data.migrations import Migration, run_migrations


def _make_conn(tmp_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        tmp_path / "test.db",
        check_same_thread=False,
        isolation_level=None,
    )


def test_runs_all_migrations_on_fresh_db(tmp_path: Path) -> None:
    conn = _make_conn(tmp_path)
    calls: list[str] = []

    def v1(c: sqlite3.Connection) -> None:
        calls.append("v1")
        c.execute("CREATE TABLE foo (id INTEGER)")

    def v2(c: sqlite3.Connection) -> None:
        calls.append("v2")
        c.execute("CREATE TABLE bar (id INTEGER)")

    applied = run_migrations(
        conn,
        [
            Migration(version=1, name="foo_init", apply=v1),
            Migration(version=2, name="bar_init", apply=v2),
        ],
        namespace="test",
    )
    assert applied == [1, 2]
    assert calls == ["v1", "v2"]
    # Both tables exist.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "foo" in tables
    assert "bar" in tables


def test_is_idempotent_on_second_run(tmp_path: Path) -> None:
    conn = _make_conn(tmp_path)
    invocations = {"count": 0}

    def v1(c: sqlite3.Connection) -> None:
        invocations["count"] += 1
        c.execute("CREATE TABLE IF NOT EXISTS foo (id INTEGER)")

    migrations = [Migration(version=1, name="init", apply=v1)]
    first = run_migrations(conn, migrations, namespace="test")
    second = run_migrations(conn, migrations, namespace="test")
    assert first == [1]
    assert second == []  # already applied
    assert invocations["count"] == 1


def test_appends_only_new_migrations(tmp_path: Path) -> None:
    conn = _make_conn(tmp_path)

    def v1(c: sqlite3.Connection) -> None:
        c.execute("CREATE TABLE foo (id INTEGER)")

    def v2(c: sqlite3.Connection) -> None:
        c.execute("CREATE TABLE bar (id INTEGER)")

    run_migrations(
        conn, [Migration(version=1, name="init", apply=v1)], namespace="test"
    )
    # Operator ships a new release adding v2.
    applied = run_migrations(
        conn,
        [
            Migration(version=1, name="init", apply=v1),
            Migration(version=2, name="add_bar", apply=v2),
        ],
        namespace="test",
    )
    assert applied == [2]


def test_rejects_duplicate_versions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate migration version"):
        run_migrations(
            _make_conn(tmp_path),
            [
                Migration(version=1, name="a", apply=lambda c: None),
                Migration(version=1, name="b", apply=lambda c: None),
            ],
            namespace="test",
        )


def test_rejects_out_of_order_versions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        run_migrations(
            _make_conn(tmp_path),
            [
                Migration(version=2, name="b", apply=lambda c: None),
                Migration(version=1, name="a", apply=lambda c: None),
            ],
            namespace="test",
        )


def test_failed_migration_does_not_record_version(tmp_path: Path) -> None:
    """A thrown apply leaves the version unrecorded so the next run retries.

    We deliberately do NOT wrap the apply in an explicit transaction
    (see the module docstring) — SQLite's autocommit + executescript
    semantics make external rollback unreliable. Instead we rely on
    migrations being idempotent so the re-try is a no-op for any
    side effect that already landed.
    """
    conn = _make_conn(tmp_path)

    def v1(c: sqlite3.Connection) -> None:
        c.execute("CREATE TABLE foo (id INTEGER)")

    def v2(c: sqlite3.Connection) -> None:
        # Partial side effect lands; then we raise. The version row
        # must NOT be recorded so the next run picks this up again.
        c.execute("CREATE TABLE bar (id INTEGER)")
        raise RuntimeError("simulated migration failure")

    with pytest.raises(RuntimeError, match="simulated"):
        run_migrations(
            conn,
            [
                Migration(version=1, name="init", apply=v1),
                Migration(version=2, name="broken", apply=v2),
            ],
            namespace="test",
        )
    versions = {
        row[0] for row in conn.execute("SELECT version FROM _schema_migrations")
    }
    assert versions == {1}, "v2 must not be recorded on apply failure"


def test_failed_then_retried_migration_can_recover(tmp_path: Path) -> None:
    """The re-try path: fix the migration, re-run, version lands."""
    conn = _make_conn(tmp_path)
    attempts = {"count": 0}

    def v1_first_try_fails(c: sqlite3.Connection) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("first try fails")
        c.execute("CREATE TABLE foo (id INTEGER)")

    migrations = [Migration(version=1, name="init", apply=v1_first_try_fails)]
    with pytest.raises(RuntimeError):
        run_migrations(conn, migrations, namespace="test")
    # Operator re-runs (or the process restarts) — same migration, but
    # this time apply succeeds.
    applied = run_migrations(conn, migrations, namespace="test")
    assert applied == [1]


def test_user_store_creates_schema_via_migrations(tmp_path: Path) -> None:
    """Smoke test: user_store actually applies its v1 migration on first use."""
    from src.data.user_store import UserStore

    store = UserStore(tmp_path / "users.db")
    # Triggers _connect → run_migrations.
    store.create_user("a@b.com", "longenough")
    conn = sqlite3.connect(tmp_path / "users.db")
    versions = {
        row[0]
        for row in conn.execute(
            "SELECT version FROM _schema_migrations WHERE namespace = ?",
            ("user_store",),
        )
    }
    assert 1 in versions
    conn.close()
