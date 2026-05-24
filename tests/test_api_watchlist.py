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
# Add symbol
# ---------------------------------------------------------------------------


def test_post_adds_symbol_and_reflects_in_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, controller = _build_client(tmp_path, default_seed=[])
    _signup(client)
    res = client.post("/api/watchlist/symbols", json={"symbol": "nvda"})
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
    client.post("/api/watchlist/symbols", json={"symbol": "NVDA"})
    res = client.post("/api/watchlist/symbols", json={"symbol": "nvda"})
    assert res.status_code == 409


def test_post_rejects_malformed_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    # Pydantic shape validator (alphanumeric + . / -) rejects.
    res = client.post("/api/watchlist/symbols", json={"symbol": "NV DA"})
    assert res.status_code == 422


def test_post_accepts_dotted_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    res = client.post("/api/watchlist/symbols", json={"symbol": "BRK.B"})
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
    client.post("/api/watchlist/symbols", json={"symbol": "NVDA"})
    res = client.delete("/api/watchlist/symbols/NVDA")
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
    res = client.delete("/api/watchlist/symbols/NVDA")
    assert res.status_code == 200
    assert res.json() == {"removed": False, "symbol": "NVDA"}


def test_delete_normalizes_symbol_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, default_seed=[])
    _signup(client)
    client.post("/api/watchlist/symbols", json={"symbol": "NVDA"})
    res = client.delete("/api/watchlist/symbols/nvda")
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
    client_a.post("/api/watchlist/symbols", json={"symbol": "AAPL"})

    # New TestClient = new cookie jar = new browser session. Reuses
    # the underlying app (and DB) so user B is created in the same
    # store.
    client_b = TestClient(client_a.app)
    _signup(client_b, email="b@user.com")
    client_b.post("/api/watchlist/symbols", json={"symbol": "MSFT"})

    a_syms = client_a.get("/api/watchlist").json()["symbols"]
    b_syms = client_b.get("/api/watchlist").json()["symbols"]
    assert a_syms == ["AAPL"]
    assert b_syms == ["MSFT"]
