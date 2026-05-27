"""Persistent user accounts + session token store.

SQLite-backed, lazy-init, mirrors the established :class:`HealthStore`
pattern. Four tables:

* ``users`` — email + bcrypt-hashed password + ``email_verified`` flag
  + audit timestamps.
* ``sessions`` — opaque session tokens with TTL, foreign-keyed to a
  user via ``user_id``.
* ``password_reset_tokens`` — single-use reset tokens with TTL. Marked
  ``used_at`` on first redeem so a replay attempt fails closed.
* ``email_verification_tokens`` — single-use verification tokens with
  TTL. Same shape as password reset tokens; consume flips the user's
  ``email_verified`` bit and marks the token used in one transaction.

Sessions are random 32-byte tokens stored verbatim — they live in
HTTP-only cookies signed with the app's session secret. The
combination of cookie signing (proves the token came from us) and
DB lookup (proves the token is still valid) keeps revocation cheap:
delete the row, the token stops working immediately even if the
browser still has it.

Why bcrypt + a separate DB
--------------------------
* bcrypt over plain SHA-2 because passwords. Adaptive cost factor
  (12 rounds default) keeps brute-force economically uninteresting
  for the next decade.
* Separate ``users.db`` from ``health.db`` because user state has
  different backup + access-control needs than observability. A
  health-store wipe should never touch user accounts.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import bcrypt

from src.data.migrations import Migration, run_migrations
from src.utils.logging import get_logger

log = get_logger(__name__)


# bcrypt cost factor. 12 rounds takes ~250ms on modern hardware —
# enough to defeat brute force, fast enough to not affect login UX.
_BCRYPT_ROUNDS = 12


# Session token byte count. 32 bytes = 256 bits of entropy via
# secrets.token_urlsafe — well past the threshold for unguessable.
_SESSION_TOKEN_BYTES = 32


# v1 — initial users + sessions tables. Lifted verbatim from the
# pre-migration inline schema so existing databases re-apply cleanly
# (CREATE TABLE IF NOT EXISTS is idempotent). When future column /
# index changes land, append a v2 ``Migration`` below; do not edit v1.
_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
"""


def _apply_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_V1_SCHEMA)


# v2 — password_reset_tokens. Single-use, TTL-bounded. ``used_at`` is
# NULL until the token is consumed; once set, the same token cannot
# be redeemed again (consume_password_reset_token enforces). Indexes
# cover the two access patterns: lookup by token (PK), and the
# expiry-prune sweep (expires_at).
_V2_SCHEMA = """
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_password_reset_user
    ON password_reset_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_password_reset_expires
    ON password_reset_tokens(expires_at);
"""


def _apply_v2(conn: sqlite3.Connection) -> None:
    conn.executescript(_V2_SCHEMA)


# v3 — email verification. Adds a boolean ``email_verified`` column
# to ``users`` (default 0 for the "verified" semantic; existing rows
# default to unverified so the post-deploy banner appears for every
# pre-migration account, which is the safer fail-shut default) and a
# new ``email_verification_tokens`` table that mirrors the
# password-reset structure (single-use, TTL, FK cascade).
_V3_SCHEMA = """
ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS email_verification_tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_email_verification_user
    ON email_verification_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_email_verification_expires
    ON email_verification_tokens(expires_at);
"""


def _apply_v3(conn: sqlite3.Connection) -> None:
    conn.executescript(_V3_SCHEMA)


_MIGRATIONS = [
    Migration(version=1, name="users_sessions_init", apply=_apply_v1),
    Migration(version=2, name="password_reset_tokens", apply=_apply_v2),
    Migration(version=3, name="email_verification", apply=_apply_v3),
]
_MIGRATION_NAMESPACE = "user_store"


class EmailAlreadyRegistered(ValueError):
    """Raised when ``create_user`` is called with an email that exists."""


class InvalidCredentials(ValueError):
    """Raised by :meth:`UserStore.verify_login` on email/password mismatch."""


@dataclass(frozen=True)
class User:
    """Public-facing user record. ``password_hash`` is intentionally
    omitted — callers handle it via UserStore methods only, never
    serialize it onto the wire."""

    id: int
    email: str
    created_at: datetime
    last_login_at: datetime | None
    email_verified: bool = False

    def to_wire(self) -> dict[str, str | int | bool | None]:
        """JSON-safe shape for ``/api/auth/me`` etc. No password hash."""
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "last_login_at": (
                self.last_login_at.isoformat()
                if self.last_login_at is not None
                else None
            ),
            "email_verified": self.email_verified,
        }


@dataclass(frozen=True)
class Session:
    """One issued session token + its TTL.

    Used by the API layer to thread an authenticated user through
    each request. Construction is internal — callers either get a
    session back from :meth:`UserStore.create_session` (on login)
    or resolve one from a cookie via :meth:`UserStore.lookup_session`.
    """

    token: str
    user_id: int
    created_at: datetime
    expires_at: datetime


class UserStore:
    """SQLite-backed user + session store.

    Constructed cheaply (no I/O). Same lazy-init contract as
    :class:`HealthStore`: the DB file + schema appear on the first
    actual read/write call so test stores at bogus paths don't
    materialize anything.

    Concurrency: per-store ``threading.Lock`` serializes writes
    across FastAPI's threadpool. WAL allows concurrent readers.
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
        # Versioned migrations. v1 is the historical CREATE TABLE IF
        # NOT EXISTS block — applies cleanly to brand-new files and is
        # a no-op for files created before the migration runner existed
        # (the _schema_migrations row will be recorded on first run,
        # and subsequent connects skip it). See src/data/migrations.py.
        run_migrations(conn, _MIGRATIONS, namespace=_MIGRATION_NAMESPACE)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----- user CRUD --------------------------------------------------

    def create_user(self, email: str, password: str) -> User:
        """Create a new user. Raises :class:`EmailAlreadyRegistered` on dupe.

        Password is hashed with bcrypt at the configured cost factor
        before insert; the plaintext never persists. Email is stored
        lower-cased + trimmed — matches the COLLATE NOCASE on the
        unique index so a re-signup with mixed case is correctly
        rejected.
        """
        email_norm = _normalize_email(email)
        if not email_norm:
            raise ValueError("email must not be empty")
        if not password:
            raise ValueError("password must not be empty")
        hashed = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
        )
        created_at = datetime.now(UTC)
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN")
            try:
                cur = conn.execute(
                    """
                    INSERT INTO users (email, password_hash, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (email_norm, hashed.decode("utf-8"), created_at.isoformat()),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                if "users.email" in str(exc).lower():
                    raise EmailAlreadyRegistered(email_norm) from None
                raise
        return User(
            id=int(cur.lastrowid or 0),
            email=email_norm,
            created_at=created_at,
            last_login_at=None,
        )

    def get_user_by_id(self, user_id: int) -> User | None:
        conn = self._connect()
        cur = conn.execute(
            "SELECT id, email, created_at, last_login_at, email_verified "
            "FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_user(row)

    def get_user_by_email(self, email: str) -> User | None:
        email_norm = _normalize_email(email)
        if not email_norm:
            return None
        conn = self._connect()
        cur = conn.execute(
            "SELECT id, email, created_at, last_login_at, email_verified "
            "FROM users WHERE email = ?",
            (email_norm,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_user(row)

    def delete_user(self, user_id: int) -> bool:
        """Remove ``user_id`` and (via ON DELETE CASCADE) their sessions.

        Returns True iff a row was deleted; False when no such user
        exists (idempotent — safe to call from a rollback path that
        races with another deletion). Used by the signup-seed atomicity
        path (B-14): if the post-signup hook fails after the user is
        created, this restores the "user account does not exist" state
        so the caller can retry signup cleanly.

        Watchlist rows live in :class:`WatchlistStore` (separate
        connection, same DB file) and intentionally have no FK to
        ``users(id)`` — see the comment in ``watchlist_store.py``.
        The caller must purge those explicitly when needed; the only
        site that does this today is the signup-seed rollback, which
        owns the WatchlistStore handle directly.
        """
        with self._lock:
            conn = self._connect()
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cur.rowcount > 0

    def verify_login(self, email: str, password: str) -> User:
        """Return the user if email+password match. Else :class:`InvalidCredentials`.

        Constant-time-ish via bcrypt's checkpw (the dominant cost is
        bcrypt itself, not the comparison). Email-not-found also
        runs a dummy hash to avoid leaking which emails are
        registered via timing.
        """
        email_norm = _normalize_email(email)
        conn = self._connect()
        cur = conn.execute(
            "SELECT id, email, password_hash, created_at, last_login_at, "
            "email_verified FROM users WHERE email = ?",
            (email_norm,),
        )
        row = cur.fetchone()
        if row is None:
            # Burn an equivalent bcrypt op against a dummy hash so the
            # signal from no-such-user vs wrong-password is identical
            # at the timing layer.
            bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
            raise InvalidCredentials("invalid email or password")
        if not bcrypt.checkpw(
            password.encode("utf-8"), row[2].encode("utf-8")
        ):
            raise InvalidCredentials("invalid email or password")
        # Record last_login for audit. Failure here must not abort
        # login — wrap defensively.
        now = datetime.now(UTC)
        try:
            with self._lock:
                conn.execute(
                    "UPDATE users SET last_login_at = ? WHERE id = ?",
                    (now.isoformat(), row[0]),
                )
        except sqlite3.Error as exc:
            log.warning(
                "user_store.last_login.update_failed",
                user_id=row[0],
                error=str(exc),
            )
        return User(
            id=int(row[0]),
            email=row[1],
            created_at=_parse_iso(row[3]),
            last_login_at=now,
            email_verified=bool(row[5]),
        )

    # ----- session CRUD -----------------------------------------------

    def create_session(self, user_id: int, ttl: timedelta) -> Session:
        """Mint + persist a new session token for ``user_id``.

        Token is `secrets.token_urlsafe(32)` — cryptographically
        random, URL-safe so it works inside a cookie without
        escaping. The caller is responsible for signing the cookie
        before sending it to the browser.
        """
        if ttl <= timedelta(0):
            raise ValueError(f"ttl must be positive, got {ttl!r}")
        token = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
        now = datetime.now(UTC)
        expires_at = now + ttl
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, user_id, now.isoformat(), expires_at.isoformat()),
            )
        return Session(
            token=token,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
        )

    def lookup_session(self, token: str) -> User | None:
        """Resolve a session token to a User. Returns None for missing
        or expired sessions; transparently prunes expired rows."""
        if not token:
            return None
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT s.user_id, s.expires_at,
                   u.id, u.email, u.created_at, u.last_login_at,
                   u.email_verified
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        expires_at = _parse_iso(row[1])
        if expires_at <= datetime.now(UTC):
            # Expired — clean up + reject. Best-effort delete; the
            # next request resolves the same way regardless.
            try:
                with self._lock:
                    conn.execute(
                        "DELETE FROM sessions WHERE token = ?", (token,)
                    )
            except sqlite3.Error as exc:
                log.warning(
                    "user_store.session.prune_failed",
                    error=str(exc),
                )
            return None
        return User(
            id=int(row[2]),
            email=row[3],
            created_at=_parse_iso(row[4]),
            last_login_at=_parse_iso(row[5]) if row[5] is not None else None,
            email_verified=bool(row[6]),
        )

    def revoke_session(self, token: str) -> bool:
        """Delete a session. Returns True iff a row was removed."""
        if not token:
            return False
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM sessions WHERE token = ?", (token,)
            )
            return cur.rowcount > 0

    def prune_expired_sessions(self) -> int:
        """Drop every session whose ``expires_at`` is in the past.

        Returns the count deleted. Intended to be called from a
        periodic worker; not on the hot path.
        """
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (now,)
            )
            return int(cur.rowcount or 0)

    # ----- password reset --------------------------------------------

    def create_password_reset_token(
        self, user_id: int, ttl: timedelta
    ) -> str:
        """Mint a single-use reset token bound to ``user_id``.

        Token is `secrets.token_urlsafe(32)` — same entropy as session
        tokens, URL-safe so it travels cleanly in a reset link query
        string. Caller (the API route) embeds the raw token in the
        emailed URL; subsequent verification goes through
        :meth:`consume_password_reset_token`.

        The caller is responsible for the "if email exists" branching
        — this method assumes the user_id is valid. The route uses
        :meth:`get_user_by_email` first and silently no-ops on
        unknown emails so the response can't be used for enumeration.
        """
        if ttl <= timedelta(0):
            raise ValueError(f"ttl must be positive, got {ttl!r}")
        token = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
        now = datetime.now(UTC)
        expires_at = now + ttl
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO password_reset_tokens
                    (token, user_id, created_at, expires_at, used_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (token, user_id, now.isoformat(), expires_at.isoformat()),
            )
        return token

    def consume_password_reset_token(self, token: str) -> User | None:
        """Validate + atomically mark a reset token used.

        Returns the bound :class:`User` if the token is valid (exists,
        unused, not yet expired); ``None`` otherwise.

        The state machine is one-way: a token transitions from
        ``unused → used`` exactly once. The UPDATE ... WHERE used_at
        IS NULL guard handles the (theoretical) concurrent-redeem
        race — the second caller sees zero affected rows and gets
        ``None``. SQLite's per-write lock makes this functionally
        atomic on our deployment shape.

        Caller follows up with :meth:`update_password` +
        :meth:`revoke_all_sessions_for_user`. We keep those as
        separate methods so the route can sequence them in the right
        order (consume → update → revoke) and log between steps.
        """
        if not token:
            return None
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """
                UPDATE password_reset_tokens
                SET used_at = ?
                WHERE token = ?
                  AND used_at IS NULL
                  AND expires_at > ?
                RETURNING user_id
                """,
                (now_iso, token, now_iso),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self.get_user_by_id(int(row[0]))

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Re-hash + persist a new password for ``user_id``.

        Returns True iff a row was updated. bcrypt cost factor matches
        :meth:`create_user` so the per-user upgrade story is uniform.
        The caller should immediately follow with
        :meth:`revoke_all_sessions_for_user` so any in-flight session
        on a now-stolen cookie is invalidated.
        """
        if not new_password:
            raise ValueError("password must not be empty")
        hashed = bcrypt.hashpw(
            new_password.encode("utf-8"),
            bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
        )
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hashed.decode("utf-8"), user_id),
            )
            return cur.rowcount > 0

    def revoke_all_sessions_for_user(self, user_id: int) -> int:
        """Delete every session row for ``user_id``. Returns count.

        Called from the password-reset confirm path so an attacker
        who held an old session cookie loses access the moment the
        password changes — defense in depth on top of the cookie
        signing + DB lookup gate.
        """
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            )
            return int(cur.rowcount or 0)

    def prune_expired_password_reset_tokens(self) -> int:
        """Drop every reset token whose ``expires_at`` is in the past.

        Symmetric with :meth:`prune_expired_sessions`. Not on the hot
        path — a periodic sweep is enough; consume already filters
        expired tokens out of the success path.
        """
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM password_reset_tokens WHERE expires_at <= ?",
                (now,),
            )
            return int(cur.rowcount or 0)

    def verify_password(self, user_id: int, password: str) -> bool:
        """Constant-time-ish password check for a known user_id.

        Distinct from :meth:`verify_login` because the caller already
        knows which user they're checking (account-delete confirmation,
        future "change password while logged in" flow). Returns False
        for missing user / wrong password — never raises so the route
        can collapse both into one generic "wrong password" response
        and not leak existence.
        """
        if not password:
            return False
        conn = self._connect()
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        )
        row = cur.fetchone()
        if row is None:
            # Burn an equivalent bcrypt op so timing doesn't
            # distinguish missing-id from wrong-password.
            bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
            return False
        return bool(
            bcrypt.checkpw(password.encode("utf-8"), row[0].encode("utf-8"))
        )

    # ----- email verification -----------------------------------------

    def create_email_verification_token(
        self, user_id: int, ttl: timedelta
    ) -> str:
        """Mint a single-use verification token bound to ``user_id``.

        Same shape as :meth:`create_password_reset_token`. The caller
        embeds the token in the emailed URL; consumption flips the
        user's ``email_verified`` bit + marks the token used in one
        atomic UPDATE.

        Idempotent at the "user has many pending tokens" level: minting
        a fresh token doesn't invalidate previous ones. They all
        remain redeemable until expired/used — useful when a user
        re-requests a verification email and the old one was already
        delivered but unread. Once any one is redeemed the user is
        verified; the others become harmless because the consume path
        is a no-op when ``email_verified`` is already True.
        """
        if ttl <= timedelta(0):
            raise ValueError(f"ttl must be positive, got {ttl!r}")
        token = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
        now = datetime.now(UTC)
        expires_at = now + ttl
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO email_verification_tokens
                    (token, user_id, created_at, expires_at, used_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (token, user_id, now.isoformat(), expires_at.isoformat()),
            )
        return token

    def consume_email_verification_token(self, token: str) -> User | None:
        """Validate, mark used, flip ``email_verified=True``.

        Atomic via a single transaction over two UPDATEs:

        1. Mark the token used IFF unused + not expired.
        2. If step 1 affected a row, flip ``email_verified`` on the
           bound user.

        Returns the verified :class:`User` on success, ``None`` for
        any invalid token state (unknown / expired / replayed). Idempotent
        on already-verified users — the second UPDATE is a no-op write
        of ``email_verified=1`` when the user was already verified.
        """
        if not token:
            return None
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        with self._lock:
            conn = self._connect()
            conn.execute("BEGIN")
            try:
                cur = conn.execute(
                    """
                    UPDATE email_verification_tokens
                    SET used_at = ?
                    WHERE token = ?
                      AND used_at IS NULL
                      AND expires_at > ?
                    RETURNING user_id
                    """,
                    (now_iso, token, now_iso),
                )
                row = cur.fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                user_id = int(row[0])
                conn.execute(
                    "UPDATE users SET email_verified = 1 WHERE id = ?",
                    (user_id,),
                )
                conn.execute("COMMIT")
            except sqlite3.Error:
                conn.execute("ROLLBACK")
                raise
        return self.get_user_by_id(user_id)

    def prune_expired_email_verification_tokens(self) -> int:
        """Sweep helper — symmetric with the other prune methods."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM email_verification_tokens WHERE expires_at <= ?",
                (now,),
            )
            return int(cur.rowcount or 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Pre-computed dummy hash used in :meth:`UserStore.verify_login` to
# burn a bcrypt op when the email doesn't exist — defeats the timing
# side-channel that would otherwise leak which addresses are
# registered. Generated once at import time; cost factor matches.
_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt(rounds=_BCRYPT_ROUNDS))


def _normalize_email(email: str) -> str:
    """Trim + lowercase. No deeper validation — let the user store
    accept anything that round-trips through the unique index."""
    return (email or "").strip().lower()


def _row_to_user(row: tuple[int, str, str, str | None, int]) -> User:
    return User(
        id=int(row[0]),
        email=row[1],
        created_at=_parse_iso(row[2]),
        last_login_at=_parse_iso(row[3]) if row[3] is not None else None,
        email_verified=bool(row[4]),
    )


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "EmailAlreadyRegistered",
    "InvalidCredentials",
    "Session",
    "User",
    "UserStore",
]
