"""Persistent user accounts + session token store.

SQLite-backed, lazy-init, mirrors the established :class:`HealthStore`
pattern. Two tables:

* ``users`` — email + bcrypt-hashed password + audit timestamps.
* ``sessions`` — opaque session tokens with TTL, foreign-keyed to a
  user via ``user_id``.

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

from src.utils.logging import get_logger

log = get_logger(__name__)


# bcrypt cost factor. 12 rounds takes ~250ms on modern hardware —
# enough to defeat brute force, fast enough to not affect login UX.
_BCRYPT_ROUNDS = 12


# Session token byte count. 32 bytes = 256 bits of entropy via
# secrets.token_urlsafe — well past the threshold for unguessable.
_SESSION_TOKEN_BYTES = 32


_SCHEMA = """
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

    def to_wire(self) -> dict[str, str | int | None]:
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
        conn.executescript(_SCHEMA)
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
            "SELECT id, email, created_at, last_login_at FROM users WHERE id = ?",
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
            "SELECT id, email, created_at, last_login_at FROM users WHERE email = ?",
            (email_norm,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_user(row)

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
            "SELECT id, email, password_hash, created_at, last_login_at "
            "FROM users WHERE email = ?",
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
                   u.id, u.email, u.created_at, u.last_login_at
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


def _row_to_user(row: tuple[int, str, str, str | None]) -> User:
    return User(
        id=int(row[0]),
        email=row[1],
        created_at=_parse_iso(row[2]),
        last_login_at=_parse_iso(row[3]) if row[3] is not None else None,
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
