"""Tests for ``/api/watchlist`` — GET, POST symbol, DELETE symbol.

End-to-end through the FastAPI TestClient. Confirms auth-required
gating, the round-trip with the WatchlistStore, the signup-time
default seed, the controller's ``add_symbol`` hook firing on POST,
and the idempotent DELETE contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController
from src.data.user_store import UserStore
from src.data.watchlist_store import WatchlistStore


def _build_client(
    tmp_path: Path, *, default_seed: list[str] | None = None
) -> tuple[TestClient, MockDashboardController]:
    """Build a TestClient + return the controller so tests can poke at it.

    The user store + watchlist store share ``users.db`` (the production
    layout). The controller starts with a small known watchlist so
    add_symbol calls from POSTs are observable as deltas. Tests pass
    ``default_seed`` to override the per-signup default — ``[]`` means
    new users land on an empty watchlist (cleanest baseline for
    POST/DELETE tests).
    """
    users = UserStore(tmp_path / "users.db")
    wl = WatchlistStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)

    app = create_app(
        controller,
        user_store=users,
        watchlist_store=wl,
        default_watchlist_symbols=default_seed,
    )
    return TestClient(app), controller


def _signup(client: TestClient, email: str = "a@b.com") -> None:
    res = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "longenough"},
    )
    assert res.status_code == 201, res.text


def _csrf(client: TestClient) -> dict[str, str]:
    """Pull the CSRF cookie off the TestClient's jar and return headers.

    Mirrors what the frontend's :func:`csrfHeaders` helper does.
    Empty dict when no cookie is present (e.g. anonymous tests).
    """
    token = client.cookies.get("esther_csrf")
    return {"X-CSRF-Token": token} if token else {}


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------


def test_get_requires_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path)
    res = client.get("/api/watchlist")
    assert res.status_code == 401


def test_post_requires_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path)
    res = client.post("/api/watchlist/symbols", json={"symbol": "NVDA"})
    assert res.status_code == 401


def test_delete_requires_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path)
    res = client.delete("/api/watchlist/symbols/NVDA")
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# Default seed on signup
# ---------------------------------------------------------------------------


def test_signup_seeds_default_watchlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, controller = _build_client(
        tmp_path, default_seed=["AAPL", "MSFT", "NVDA"]
    )
    _signup(client)
    res = client.get("/api/watchlist")
    assert res.status_code == 200
    assert res.json()["symbols"] == ["AAPL", "MSFT", "NVDA"]
    # Controller must be tracking the seeded symbols so the next
    # snapshot tick produces rows for them.
    assert set(controller.watchlist) >= {"AAPL", "MSFT", "NVDA"}


def test_signup_with_empty_default_seed_yields_empty_watchlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    res = client.get("/api/watchlist")
    assert res.status_code == 200
    assert res.json()["symbols"] == []


# ---------------------------------------------------------------------------
# B-14: transactional signup — seed failure rolls back the user.
# ---------------------------------------------------------------------------


class _FailingWatchlistStore(WatchlistStore):
    """WatchlistStore whose ``seed_default`` always raises.

    Simulates a transient backend failure (e.g., SQLite lock) on the
    seed step. The signup handler must surface a 5xx and roll back the
    user so the trader can retry cleanly.
    """

    def seed_default(self, user_id: int, symbols):  # type: ignore[no-untyped-def, override]
        raise RuntimeError("simulated seed failure")


def _build_client_with_failing_seed(
    tmp_path: Path,
) -> tuple[TestClient, UserStore]:
    users = UserStore(tmp_path / "users.db")
    wl = _FailingWatchlistStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        user_store=users,
        watchlist_store=wl,
        default_watchlist_symbols=["AAPL", "MSFT"],
    )
    return TestClient(app), users


def test_signup_with_failing_seed_returns_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hook failure aborts signup with a service-unavailable status —
    no more silent 201 + empty dashboard (B-14)."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client_with_failing_seed(tmp_path)
    res = client.post(
        "/api/auth/signup",
        json={"email": "victim@user.com", "password": "longenough"},
    )
    assert 500 <= res.status_code < 600, res.text
    # No session cookie should have been issued.
    assert "set-cookie" not in {h.lower() for h in res.headers}


def test_signup_with_failing_seed_rolls_back_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a failed seed the user row must not persist — otherwise a
    retry with the same email would 409 forever and the trader is
    permanently locked out."""
    monkeypatch.chdir(tmp_path)
    client, users = _build_client_with_failing_seed(tmp_path)
    client.post(
        "/api/auth/signup",
        json={"email": "victim@user.com", "password": "longenough"},
    )
    # Direct store lookup: the user must not exist post-rollback.
    assert users.get_user_by_email("victim@user.com") is None


def test_signup_rollback_lets_user_retry_with_same_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a transient seed failure followed by a retry
    against a recovered backend produces a clean signup."""
    monkeypatch.chdir(tmp_path)
    users = UserStore(tmp_path / "users.db")
    wl = WatchlistStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        user_store=users,
        watchlist_store=wl,
        default_watchlist_symbols=["AAPL"],
    )
    client = TestClient(app)

    # Simulate one transient failure by patching seed_default to raise once.
    original_seed = wl.seed_default
    calls = {"n": 0}

    def flaky_seed(user_id: int, symbols):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient SQLite lock")
        return original_seed(user_id, symbols)

    wl.seed_default = flaky_seed  # type: ignore[method-assign]
    first = client.post(
        "/api/auth/signup",
        json={"email": "retry@user.com", "password": "longenough"},
    )
    assert 500 <= first.status_code < 600
    # Retry succeeds against the recovered backend, same email.
    second = client.post(
        "/api/auth/signup",
        json={"email": "retry@user.com", "password": "longenough"},
    )
    assert second.status_code == 201, second.text
    me = client.get("/api/auth/me")
    assert me.json()["user"]["email"] == "retry@user.com"


# ---------------------------------------------------------------------------
# Add symbol
# ---------------------------------------------------------------------------


def test_post_adds_symbol_and_reflects_in_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, controller = _build_client(tmp_path, default_seed=[])
    _signup(client)
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "nvda"},
        headers=_csrf(client),
    )
    assert res.status_code == 201
    assert res.json()["entry"]["symbol"] == "NVDA"
    # GET reflects it.
    listing = client.get("/api/watchlist").json()
    assert listing["symbols"] == ["NVDA"]
    # And the controller now tracks NVDA so the snapshot pipeline
    # produces a row for it on the next tick.
    assert "NVDA" in controller.watchlist


def test_post_duplicate_returns_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NVDA"},
        headers=_csrf(client),
    )
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "nvda"},
        headers=_csrf(client),
    )
    assert res.status_code == 409


def test_post_rejects_malformed_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    # Pydantic shape validator (alphanumeric + . / -) rejects.
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NV DA"},
        headers=_csrf(client),
    )
    assert res.status_code == 422


def test_post_accepts_dotted_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "BRK.B"},
        headers=_csrf(client),
    )
    assert res.status_code == 201
    assert res.json()["entry"]["symbol"] == "BRK.B"


# ---------------------------------------------------------------------------
# Remove symbol
# ---------------------------------------------------------------------------


def test_delete_removes_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NVDA"},
        headers=_csrf(client),
    )
    res = client.delete(
        "/api/watchlist/symbols/NVDA",
        headers=_csrf(client),
    )
    assert res.status_code == 200
    body = res.json()
    assert body == {"removed": True, "symbol": "NVDA"}
    assert client.get("/api/watchlist").json()["symbols"] == []


def test_delete_idempotent_on_missing_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing a symbol the user never had returns ``removed: false``,
    not 404. The desired state (symbol absent) is what the caller wanted."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    res = client.delete(
        "/api/watchlist/symbols/NVDA",
        headers=_csrf(client),
    )
    assert res.status_code == 200
    assert res.json() == {"removed": False, "symbol": "NVDA"}


def test_delete_normalizes_symbol_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NVDA"},
        headers=_csrf(client),
    )
    res = client.delete(
        "/api/watchlist/symbols/nvda",
        headers=_csrf(client),
    )
    assert res.status_code == 200
    assert res.json()["removed"] is True


# ---------------------------------------------------------------------------
# Per-user isolation
# ---------------------------------------------------------------------------


def test_two_users_have_independent_watchlists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each user's mutations are confined to their own list."""
    monkeypatch.chdir(tmp_path)
    client_a, _ = _build_client(tmp_path, default_seed=[])
    _signup(client_a, email="a@user.com")
    client_a.post(
        "/api/watchlist/symbols",
        json={"symbol": "AAPL"},
        headers=_csrf(client_a),
    )

    # New TestClient = new cookie jar = new browser session. Reuses
    # the underlying app (and DB) so user B is created in the same
    # store.
    client_b = TestClient(client_a.app)
    _signup(client_b, email="b@user.com")
    client_b.post(
        "/api/watchlist/symbols",
        json={"symbol": "MSFT"},
        headers=_csrf(client_b),
    )

    a_syms = client_a.get("/api/watchlist").json()["symbols"]
    b_syms = client_b.get("/api/watchlist").json()["symbols"]
    assert a_syms == ["AAPL"]
    assert b_syms == ["MSFT"]
