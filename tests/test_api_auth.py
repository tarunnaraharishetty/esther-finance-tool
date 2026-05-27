"""Tests for /api/auth/* — signup, login, logout, me + session cookies.

End-to-end through the FastAPI TestClient which round-trips cookies
the same way a browser would. Confirms the signed cookie machinery,
the 401/409 error paths, and the session-resolver wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.data.user_store import UserStore


def _build_client(tmp_path: Path) -> TestClient:
    store = UserStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, user_store=store)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------


def test_signup_creates_user_and_sets_session_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.post(
        "/api/auth/signup",
        json={"email": "new@user.com", "password": "longenough"},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["user"]["email"] == "new@user.com"
    assert "id" in body["user"]
    # The TestClient persists the cookie jar on the client object; the
    # set-cookie header is visible on the response too.
    assert "set-cookie" in {h.lower() for h in res.headers}
    # A follow-up /me reuses the jar and resolves to the new user.
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "new@user.com"


def test_signup_rejects_duplicate_email_with_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    payload = {"email": "dup@user.com", "password": "longenough"}
    client.post("/api/auth/signup", json=payload)
    res = client.post("/api/auth/signup", json=payload)
    assert res.status_code == 409
    assert "already exists" in res.json()["detail"].lower()


def test_signup_rejects_short_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.post(
        "/api/auth/signup",
        json={"email": "a@b.com", "password": "short"},
    )
    # FastAPI returns 422 on pydantic validation failure.
    assert res.status_code == 422


def test_signup_rejects_malformed_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.post(
        "/api/auth/signup",
        json={"email": "no-at-sign", "password": "longenough"},
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def test_login_with_correct_credentials_sets_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={"email": "a@b.com", "password": "longenough"},
    )
    # Clear the cookie jar from signup — we want to confirm login
    # works on a fresh client too.
    client.cookies.clear()
    res = client.post(
        "/api/auth/login",
        json={"email": "a@b.com", "password": "longenough"},
    )
    assert res.status_code == 200
    assert res.json()["user"]["email"] == "a@b.com"
    me = client.get("/api/auth/me")
    assert me.json()["user"]["email"] == "a@b.com"


def test_login_with_wrong_password_returns_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={"email": "a@b.com", "password": "longenough"},
    )
    client.cookies.clear()
    res = client.post(
        "/api/auth/login",
        json={"email": "a@b.com", "password": "wrongguess"},
    )
    assert res.status_code == 401
    # Generic error message — must not leak which half (email vs pw) wrong.
    assert "invalid" in res.json()["detail"].lower()


def test_login_with_unknown_email_returns_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.post(
        "/api/auth/login",
        json={"email": "nobody@b.com", "password": "longenough"},
    )
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------


def test_me_returns_null_when_anonymous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.get("/api/auth/me")
    assert res.status_code == 200
    assert res.json() == {"user": None}


def test_me_returns_user_after_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={"email": "a@b.com", "password": "longenough"},
    )
    res = client.get("/api/auth/me")
    assert res.status_code == 200
    body = res.json()
    assert body["user"]["email"] == "a@b.com"
    # Documented per-row wire shape.
    assert set(body["user"].keys()) == {
        "id",
        "email",
        "created_at",
        "last_login_at",
        "email_verified",
    }


def test_me_with_tampered_cookie_returns_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cookie whose signature doesn't verify is rejected as anonymous."""
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    # Inject a non-signed value into the session cookie.
    client.cookies.set("esther_session", "not-a-signed-token")
    res = client.get("/api/auth/me")
    assert res.status_code == 200
    assert res.json()["user"] is None


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


def test_logout_clears_session_and_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    client.post(
        "/api/auth/signup",
        json={"email": "a@b.com", "password": "longenough"},
    )
    # Signed in
    assert client.get("/api/auth/me").json()["user"] is not None
    # Logout
    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert logout.json() == {"ok": True}
    # Browser-side: cookie gets cleared via Set-Cookie max-age=0.
    # TestClient mirrors that — the next /me is anonymous.
    assert client.get("/api/auth/me").json()["user"] is None


def test_logout_when_already_anonymous_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Logging out without a session is not an error."""
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.post("/api/auth/logout")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


# Disabled-mode behavior (USER_STORE_PATH= empty in env) is documented
# in create_app + register_auth_routes. Not asserted here — monkeypatching
# a pydantic-settings class attribute requires more gymnastics than the
# behavior is worth covering at this layer.


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


def _signup(client: TestClient, email: str = "u@e.com", pw: str = "oldpw1234") -> None:
    res = client.post(
        "/api/auth/signup", json={"email": email, "password": pw}
    )
    assert res.status_code == 201, res.text
    client.cookies.clear()  # drop the auto-login session, reset flow starts logged out


def _captured_reset_links(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Pull every reset URL the LogPasswordResetEmailer emitted."""
    return [
        getattr(rec, "reset_url", "")
        for rec in caplog.records
        if "auth.password_reset.link" in rec.getMessage()
    ]


def test_password_reset_request_returns_uniform_message_for_unknown_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-email-enumeration: the request route must return the same
    200 body whether or not the address is registered."""
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.post(
        "/api/auth/password-reset/request",
        json={"email": "nobody@example.com"},
    )
    assert res.status_code == 200
    assert res.json() == {
        "ok": True,
        "message": "If that email is registered, a reset link is on its way.",
    }


def test_password_reset_request_for_registered_email_returns_same_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    _signup(client, "u@e.com", "oldpw1234")
    res = client.post(
        "/api/auth/password-reset/request",
        json={"email": "u@e.com"},
    )
    assert res.status_code == 200
    assert res.json()["message"].startswith("If that email is registered")


def test_password_reset_end_to_end_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path through the whole flow:

    1. Signup with old password.
    2. Hit /password-reset/request — emailer fires, log emitter records
       the reset URL.
    3. Extract the token from the recorded URL.
    4. Hit /password-reset/confirm with the token + new password.
    5. Old password rejected, new password accepted on login.
    """
    monkeypatch.chdir(tmp_path)
    # Direct UserStore access so we can pull the token out without
    # relying on log inspection (which the runtime supports but the
    # TestClient log fixture doesn't always capture). The flow is
    # otherwise unchanged.
    store = UserStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, user_store=store)
    client = TestClient(app)

    # 1. Signup
    res = client.post(
        "/api/auth/signup", json={"email": "u@e.com", "password": "oldpw1234"}
    )
    assert res.status_code == 201
    client.cookies.clear()

    # 2-3. Request the reset + pull the token from the store directly
    res = client.post(
        "/api/auth/password-reset/request", json={"email": "u@e.com"}
    )
    assert res.status_code == 200

    # Fetch the active token. There's no public list method, but we
    # can introspect via consume — except consume burns the token.
    # Instead grab it from the raw connection — fair game for a test
    # that owns the store fixture.
    conn = store._connect()  # test reaches into the store on purpose
    row = conn.execute(
        "SELECT token FROM password_reset_tokens WHERE used_at IS NULL"
    ).fetchone()
    assert row is not None, "expected a freshly minted token row"
    token = row[0]

    # 4. Confirm
    res = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "newpw5678"},
    )
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    # 5. Old password rejected, new accepted
    res = client.post(
        "/api/auth/login", json={"email": "u@e.com", "password": "oldpw1234"}
    )
    assert res.status_code == 401
    res = client.post(
        "/api/auth/login", json={"email": "u@e.com", "password": "newpw5678"}
    )
    assert res.status_code == 200


def test_password_reset_confirm_rejects_invalid_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any token the store can't redeem (unknown / expired / replayed)
    collapses to one generic 400 — surfacing the exact state would
    let an attacker enumerate the token state machine."""
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    _signup(client)
    res = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": "definitely-not-a-real-token-aaaaaaaaaaaaa", "new_password": "newpw5678"},
    )
    assert res.status_code == 400
    assert "invalid or has expired" in res.json()["detail"]


def test_password_reset_confirm_revokes_existing_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user who is logged in when the reset confirms must lose
    that session — defense in depth on top of the cookie sign +
    DB lookup gate. After confirm, /api/auth/me on the cached
    session jar returns user=null."""
    monkeypatch.chdir(tmp_path)
    store = UserStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, user_store=store)
    client = TestClient(app)

    # Sign up → already logged in (post-signup mints a session).
    res = client.post(
        "/api/auth/signup",
        json={"email": "u@e.com", "password": "oldpw1234"},
    )
    assert res.status_code == 201
    # Same jar — /me works.
    assert client.get("/api/auth/me").json()["user"] is not None

    # Pull a reset token directly from the store.
    user = store.get_user_by_email("u@e.com")
    assert user is not None
    from datetime import timedelta

    token = store.create_password_reset_token(user.id, timedelta(hours=1))

    # Confirm reset — the live session must die.
    res = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "newpw5678"},
    )
    assert res.status_code == 200
    # Same jar (which the confirm response also cleared) — anonymous.
    assert client.get("/api/auth/me").json()["user"] is None


def test_password_reset_token_is_single_use_via_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror of the UserStore single-use test, exercised through the
    full route stack — protects the route from regressing
    independently of the store contract."""
    monkeypatch.chdir(tmp_path)
    store = UserStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, user_store=store)
    client = TestClient(app)

    res = client.post(
        "/api/auth/signup", json={"email": "u@e.com", "password": "oldpw1234"}
    )
    assert res.status_code == 201
    user = store.get_user_by_email("u@e.com")
    assert user is not None
    from datetime import timedelta

    token = store.create_password_reset_token(user.id, timedelta(hours=1))

    first = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "newpw5678"},
    )
    assert first.status_code == 200
    # Replay same token — must 400, just like an unknown token would.
    second = client.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "differentpw9"},
    )
    assert second.status_code == 400
