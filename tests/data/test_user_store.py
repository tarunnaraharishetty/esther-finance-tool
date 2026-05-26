"""Tests for the user + session store.

Covers the lazy-init contract, password hash + verify, duplicate-
email rejection, session create/lookup/revoke/expiry, prune
expired, and the timing-safety dummy-hash path on a missing email.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import pytest

from src.data.user_store import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    UserStore,
)


# ---------------------------------------------------------------------------
# Construction + lazy init
# ---------------------------------------------------------------------------


def test_construction_does_not_touch_disk(tmp_path: Path) -> None:
    """Critical invariant: building a UserStore is free."""
    store = UserStore(tmp_path / "missing" / "users.db")
    assert not (tmp_path / "missing").exists()
    store.close()


def test_create_user_materializes_db_and_schema(tmp_path: Path) -> None:
    db = tmp_path / "users.db"
    store = UserStore(db)
    user = store.create_user("a@b.com", "longenough")
    assert db.exists()
    assert user.id > 0
    assert user.email == "a@b.com"
    store.close()


# ---------------------------------------------------------------------------
# Password hashing + verification
# ---------------------------------------------------------------------------


def test_password_round_trips_via_verify_login(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    created = store.create_user("a@b.com", "longenough")
    logged_in = store.verify_login("a@b.com", "longenough")
    assert logged_in.id == created.id
    # last_login_at is populated on successful verify.
    assert logged_in.last_login_at is not None
    store.close()


def test_wrong_password_raises_invalid_credentials(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")
    with pytest.raises(InvalidCredentials):
        store.verify_login("a@b.com", "wrongguess")
    store.close()


def test_unknown_email_raises_invalid_credentials(tmp_path: Path) -> None:
    """Same exception as wrong-password — no leak of registered-email signal."""
    store = UserStore(tmp_path / "users.db")
    with pytest.raises(InvalidCredentials):
        store.verify_login("nobody@b.com", "doesnt-matter")
    store.close()


def test_email_lookup_is_case_insensitive(tmp_path: Path) -> None:
    """COLLATE NOCASE on the unique index means MixedCase round-trips."""
    store = UserStore(tmp_path / "users.db")
    store.create_user("Mixed@Case.com", "longenough")
    # Login with different casing must succeed.
    user = store.verify_login("MIXED@case.com", "longenough")
    assert user.email == "mixed@case.com"
    store.close()


def test_duplicate_email_raises(tmp_path: Path) -> None:
    """Index is COLLATE NOCASE — two case variants count as duplicates."""
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")
    with pytest.raises(EmailAlreadyRegistered):
        store.create_user("A@B.COM", "different-password")
    store.close()


def test_empty_email_or_password_rejected(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    with pytest.raises(ValueError):
        store.create_user("", "longenough")
    with pytest.raises(ValueError):
        store.create_user("a@b.com", "")
    store.close()


# ---------------------------------------------------------------------------
# User lookups
# ---------------------------------------------------------------------------


def test_get_user_by_id_returns_record(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    created = store.create_user("a@b.com", "longenough")
    fetched = store.get_user_by_id(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    store.close()


def test_get_user_by_id_missing_returns_none(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    assert store.get_user_by_id(9999) is None
    store.close()


def test_get_user_by_email_normalizes(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")
    assert store.get_user_by_email("  A@B.COM  ") is not None
    store.close()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_create_session_returns_random_token(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    s1 = store.create_session(user.id, timedelta(days=1))
    s2 = store.create_session(user.id, timedelta(days=1))
    assert s1.token != s2.token
    assert len(s1.token) >= 32  # url-safe encoding of 32 random bytes
    store.close()


def test_lookup_session_resolves_to_user(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    session = store.create_session(user.id, timedelta(days=1))
    resolved = store.lookup_session(session.token)
    assert resolved is not None
    assert resolved.id == user.id
    store.close()


def test_lookup_session_missing_token_returns_none(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    assert store.lookup_session("not-a-real-token") is None
    assert store.lookup_session("") is None
    store.close()


def test_revoke_session_invalidates_token(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    session = store.create_session(user.id, timedelta(days=1))
    assert store.revoke_session(session.token) is True
    assert store.lookup_session(session.token) is None
    # Re-revoking a missing token is a no-op (False).
    assert store.revoke_session(session.token) is False
    store.close()


def test_expired_session_lookup_returns_none_and_prunes(tmp_path: Path) -> None:
    """A session past its TTL resolves as None on the next lookup AND
    gets cleaned out of the table."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    # Mint a session that expires immediately.
    session = store.create_session(user.id, timedelta(milliseconds=1))
    time.sleep(0.01)
    assert store.lookup_session(session.token) is None
    # Pruning the same expired token returns 0 because lookup already cleaned it.
    assert store.prune_expired_sessions() == 0
    store.close()


def test_prune_expired_drops_only_expired(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    fresh = store.create_session(user.id, timedelta(days=1))
    expired = store.create_session(user.id, timedelta(milliseconds=1))
    time.sleep(0.01)
    pruned = store.prune_expired_sessions()
    assert pruned == 1
    # Fresh session still resolves.
    assert store.lookup_session(fresh.token) is not None
    assert store.lookup_session(expired.token) is None
    store.close()


def test_create_session_rejects_non_positive_ttl(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    with pytest.raises(ValueError):
        store.create_session(user.id, timedelta(0))
    store.close()


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def test_user_to_wire_omits_password_hash(tmp_path: Path) -> None:
    """Critical: wire shape must never carry the password hash."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    wire = user.to_wire()
    assert "password" not in wire
    assert "password_hash" not in wire
    assert set(wire.keys()) == {"id", "email", "created_at", "last_login_at"}
    store.close()


# ---------------------------------------------------------------------------
# delete_user — used by the signup-seed rollback path (B-14)
# ---------------------------------------------------------------------------


def test_delete_user_removes_row(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    assert store.delete_user(user.id) is True
    assert store.get_user_by_id(user.id) is None
    assert store.get_user_by_email("a@b.com") is None
    store.close()


def test_delete_user_missing_is_idempotent(tmp_path: Path) -> None:
    """Rollback may race with another deletion — must not raise."""
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")  # force schema init
    assert store.delete_user(99_999) is False
    store.close()


def test_delete_user_cascades_sessions(tmp_path: Path) -> None:
    """FK ON DELETE CASCADE on sessions.user_id — deleting the user
    must invalidate every issued session token in one step."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    session = store.create_session(user.id, timedelta(days=1))
    assert store.lookup_session(session.token) is not None
    store.delete_user(user.id)
    assert store.lookup_session(session.token) is None
    store.close()
