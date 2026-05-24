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
