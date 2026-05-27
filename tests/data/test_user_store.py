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
    assert set(wire.keys()) == {
        "id",
        "email",
        "created_at",
        "last_login_at",
        "email_verified",
    }
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


# ---------------------------------------------------------------------------
# Password reset tokens (migration v2)
# ---------------------------------------------------------------------------


def test_create_and_consume_password_reset_token_round_trips(
    tmp_path: Path,
) -> None:
    """Happy path: a freshly minted token resolves back to the
    bound user, and the returned User round-trips by id + email."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_password_reset_token(user.id, timedelta(hours=1))
    assert isinstance(token, str)
    assert len(token) >= 40  # secrets.token_urlsafe(32) → 43-char string
    resolved = store.consume_password_reset_token(token)
    assert resolved is not None
    assert resolved.id == user.id
    assert resolved.email == "a@b.com"
    store.close()


def test_password_reset_token_is_single_use(tmp_path: Path) -> None:
    """Critical invariant: the same token cannot be redeemed twice.

    Without this, an attacker who intercepts the reset email could
    redeem it after the legitimate user. The UPDATE ... WHERE
    used_at IS NULL guard turns the second redeem into a no-op.
    """
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_password_reset_token(user.id, timedelta(hours=1))
    assert store.consume_password_reset_token(token) is not None
    # Replay: same token, second attempt — must return None.
    assert store.consume_password_reset_token(token) is None
    store.close()


def test_password_reset_token_rejects_expired(tmp_path: Path) -> None:
    """An expired token must fail closed — the expiry filter is in
    the consume UPDATE clause."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    # Mint a token with a 1ms TTL, wait 50ms — well past expiry.
    token = store.create_password_reset_token(
        user.id, timedelta(milliseconds=1)
    )
    time.sleep(0.05)
    assert store.consume_password_reset_token(token) is None
    store.close()


def test_password_reset_token_rejects_unknown(tmp_path: Path) -> None:
    """A token never minted by us cannot be redeemed."""
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")  # force schema init
    assert store.consume_password_reset_token("not-a-real-token") is None
    assert store.consume_password_reset_token("") is None
    store.close()


def test_create_password_reset_token_rejects_non_positive_ttl(
    tmp_path: Path,
) -> None:
    """Zero or negative TTL would let a malicious caller mint a
    pre-expired token — defense in depth on the route's input
    validation."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    with pytest.raises(ValueError, match="ttl must be positive"):
        store.create_password_reset_token(user.id, timedelta(0))
    store.close()


def test_update_password_replaces_hash(tmp_path: Path) -> None:
    """After update_password, the old password fails verify_login and
    the new one succeeds — proves the hash actually rotated."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "oldpassword1")
    assert store.update_password(user.id, "newpassword1") is True
    with pytest.raises(InvalidCredentials):
        store.verify_login("a@b.com", "oldpassword1")
    assert store.verify_login("a@b.com", "newpassword1").id == user.id
    store.close()


def test_update_password_unknown_user_returns_false(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")
    assert store.update_password(99_999, "newpassword1") is False
    store.close()


def test_update_password_rejects_empty(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    with pytest.raises(ValueError, match="password must not be empty"):
        store.update_password(user.id, "")
    store.close()


def test_revoke_all_sessions_invalidates_every_token(tmp_path: Path) -> None:
    """The reset-confirm path calls this so any cookie the attacker
    held becomes useless the moment the password changes."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    s1 = store.create_session(user.id, timedelta(days=1))
    s2 = store.create_session(user.id, timedelta(days=1))
    assert store.revoke_all_sessions_for_user(user.id) == 2
    assert store.lookup_session(s1.token) is None
    assert store.lookup_session(s2.token) is None
    store.close()


def test_revoke_all_sessions_does_not_affect_other_users(
    tmp_path: Path,
) -> None:
    """Per-user scope: revoking on user A must not touch user B's
    in-flight sessions."""
    store = UserStore(tmp_path / "users.db")
    a = store.create_user("a@b.com", "longenough")
    b = store.create_user("b@c.com", "longenough")
    sa = store.create_session(a.id, timedelta(days=1))
    sb = store.create_session(b.id, timedelta(days=1))
    assert store.revoke_all_sessions_for_user(a.id) == 1
    assert store.lookup_session(sa.token) is None
    assert store.lookup_session(sb.token) is not None
    store.close()


def test_prune_expired_password_reset_tokens(tmp_path: Path) -> None:
    """Periodic-sweep helper — same shape as prune_expired_sessions."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    expired = store.create_password_reset_token(user.id, timedelta(milliseconds=1))
    live = store.create_password_reset_token(user.id, timedelta(hours=1))
    time.sleep(0.05)
    assert store.prune_expired_password_reset_tokens() == 1
    # Expired token gone; live one still resolves.
    assert store.consume_password_reset_token(expired) is None
    assert store.consume_password_reset_token(live) is not None
    store.close()


def test_password_reset_tokens_cascade_on_user_delete(tmp_path: Path) -> None:
    """FK ON DELETE CASCADE — deleting a user must wipe their pending
    reset tokens too (consistent with sessions cascade)."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_password_reset_token(user.id, timedelta(hours=1))
    store.delete_user(user.id)
    assert store.consume_password_reset_token(token) is None
    store.close()


# ---------------------------------------------------------------------------
# verify_password — re-auth for account delete + future password change
# ---------------------------------------------------------------------------


def test_verify_password_accepts_correct_password(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    assert store.verify_password(user.id, "longenough") is True
    store.close()


def test_verify_password_rejects_wrong_password(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    assert store.verify_password(user.id, "different5") is False
    store.close()


def test_verify_password_rejects_unknown_user(tmp_path: Path) -> None:
    """Missing user_id collapses to ``False`` (not an exception) so the
    route can render one generic 'wrong password' branch and not
    leak whether the id existed."""
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")  # force schema init
    assert store.verify_password(99_999, "longenough") is False
    store.close()


def test_verify_password_rejects_empty(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    assert store.verify_password(user.id, "") is False
    store.close()


# ---------------------------------------------------------------------------
# Email verification tokens (migration v3)
# ---------------------------------------------------------------------------


def test_new_user_starts_unverified(tmp_path: Path) -> None:
    """email_verified is the new ``False``-by-default column. Pre-
    deploy accounts also land here on migrate; the banner appears
    until they redeem a verification token."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    assert user.email_verified is False
    # Round-trips via every read path.
    assert store.get_user_by_id(user.id).email_verified is False  # type: ignore[union-attr]
    assert store.get_user_by_email("a@b.com").email_verified is False  # type: ignore[union-attr]
    assert store.verify_login("a@b.com", "longenough").email_verified is False
    store.close()


def test_consume_email_verification_token_flips_verified_bit(
    tmp_path: Path,
) -> None:
    """Happy path: mint a token, consume, user is now verified."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_email_verification_token(user.id, timedelta(hours=24))
    verified = store.consume_email_verification_token(token)
    assert verified is not None
    assert verified.email_verified is True
    # The flag persists — re-reading the user from any path shows it.
    assert store.get_user_by_id(user.id).email_verified is True  # type: ignore[union-attr]
    store.close()


def test_email_verification_token_is_single_use(tmp_path: Path) -> None:
    """Same shape as the password-reset replay test — the second
    redeem of the same token must return None even though the user
    is already verified."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_email_verification_token(user.id, timedelta(hours=24))
    assert store.consume_email_verification_token(token) is not None
    assert store.consume_email_verification_token(token) is None
    store.close()


def test_email_verification_token_rejects_expired(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_email_verification_token(
        user.id, timedelta(milliseconds=1)
    )
    time.sleep(0.05)
    assert store.consume_email_verification_token(token) is None
    # Failed redeem must NOT flip the verified bit.
    assert store.get_user_by_id(user.id).email_verified is False  # type: ignore[union-attr]
    store.close()


def test_email_verification_token_rejects_unknown(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    store.create_user("a@b.com", "longenough")
    assert store.consume_email_verification_token("not-real") is None
    assert store.consume_email_verification_token("") is None
    store.close()


def test_multiple_pending_verification_tokens_first_one_wins(
    tmp_path: Path,
) -> None:
    """Re-requesting a verification email doesn't invalidate the
    previous one — both remain redeemable. Whichever is redeemed
    first verifies the user; redeeming the second is a no-op (the
    user is already verified, the token's ``used_at`` gets set but
    nothing else changes)."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    t1 = store.create_email_verification_token(user.id, timedelta(hours=24))
    t2 = store.create_email_verification_token(user.id, timedelta(hours=24))
    # First redeem verifies.
    assert store.consume_email_verification_token(t1) is not None
    # Second still works (idempotent — already verified, no-op flip).
    verified = store.consume_email_verification_token(t2)
    assert verified is not None
    assert verified.email_verified is True
    store.close()


def test_email_verification_tokens_cascade_on_user_delete(
    tmp_path: Path,
) -> None:
    """FK ON DELETE CASCADE — wipe pending verification tokens too."""
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    token = store.create_email_verification_token(user.id, timedelta(hours=24))
    store.delete_user(user.id)
    assert store.consume_email_verification_token(token) is None
    store.close()


def test_prune_expired_email_verification_tokens(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.create_user("a@b.com", "longenough")
    expired = store.create_email_verification_token(
        user.id, timedelta(milliseconds=1)
    )
    live = store.create_email_verification_token(user.id, timedelta(hours=1))
    time.sleep(0.05)
    assert store.prune_expired_email_verification_tokens() == 1
    assert store.consume_email_verification_token(expired) is None
    assert store.consume_email_verification_token(live) is not None
    store.close()
