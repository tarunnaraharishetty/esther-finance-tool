"""Tests for the auth-gate middleware + per-user snapshot filter.

Two boundaries:

1. ``AuthGateMiddleware`` — when a ``UserStore`` is wired on the app,
   every ``/api/*`` route except the whitelist (``/api/health``,
   ``/api/auth/*``) must reject anonymous requests with 401.

2. ``filter_snapshot_for_symbols`` — when a user is authenticated,
   ``/api/snapshot`` returns a snapshot whose ``rows`` and
   ``ranked_opportunities`` are scoped to the user's watchlist. Other
   users' symbols never appear in the response.

We use ``TestClient`` because none of these endpoints stream; the
heavier uvicorn-in-thread fixture from ``test_api_stream`` is only
needed when verifying SSE wire format.
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
    tmp_path: Path, *, watchlist: list[str]
) -> tuple[TestClient, MockDashboardController]:
    """TestClient with auth wired + a controller covering ``watchlist``."""
    users = UserStore(tmp_path / "users.db")
    wl = WatchlistStore(tmp_path / "users.db")
    controller = MockDashboardController(watchlist=watchlist, seed=7)
    app = create_app(
        controller,
        user_store=users,
        watchlist_store=wl,
        # Default empty seed so signup doesn't pre-add symbols we'd
        # then have to subtract from the per-user filter expectations.
        default_watchlist_symbols=[],
    )
    return TestClient(app), controller


def _signup(client: TestClient, email: str = "a@b.com") -> None:
    res = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "longenough"},
    )
    assert res.status_code == 201, res.text


def _csrf(client: TestClient) -> dict[str, str]:
    """Pull the CSRF cookie off the TestClient's jar; empty when absent."""
    token = client.cookies.get("esther_csrf")
    return {"X-CSRF-Token": token} if token else {}


def _add_symbol(client: TestClient, sym: str) -> None:
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": sym},
        headers=_csrf(client),
    )
    assert res.status_code in (200, 201), res.text


# ---------------------------------------------------------------------------
# Auth gate — 401 on anonymous requests to non-whitelisted routes
# ---------------------------------------------------------------------------


def test_snapshot_requires_auth_when_user_store_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL", "MSFT"])
    res = client.get("/api/snapshot")
    assert res.status_code == 401
    assert res.json() == {"detail": "Authentication required."}


def test_health_is_public_even_with_user_store_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_readyz_returns_200_with_dependency_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/api/readyz`` is anonymous (probes from load balancers don't
    carry cookies) and surfaces per-dependency status. Confirms the
    K8s-style separation between liveness (/api/health) and readiness
    (/api/readyz) is wired."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    res = client.get("/api/readyz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    # User store is wired by _build_client; verify the probe ran.
    assert body["checks"]["user_store"] == "ok"
    # WatchlistStore probe also wired by _build_client — was missing
    # from the previous readyz so a schema break on the watchlist side
    # could hide behind a healthy users probe.
    assert body["checks"]["watchlist_store"] == "ok"


def test_livez_alias_returns_same_payload_as_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/api/livez`` exists as a k8s-convention alias for ``/api/health``.

    Kept as a separate route (rather than 301-redirecting) so a
    liveness probe never depends on the client following a redirect —
    some probe configurations treat 3xx as failure. Same payload.
    """
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    livez = client.get("/api/livez")
    health = client.get("/api/health")
    assert livez.status_code == 200
    assert livez.json() == health.json()


def test_auth_routes_remain_reachable_anonymously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    # /me explicitly returns null for anonymous — must not be 401-gated
    # itself, the SPA depends on it to decide login vs app shell.
    res = client.get("/api/auth/me")
    assert res.status_code == 200
    assert res.json() == {"user": None}


# ---------------------------------------------------------------------------
# CSRF gate — state-changing routes require the double-submit token
# ---------------------------------------------------------------------------


def test_post_without_csrf_header_is_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A logged-in user whose POST omits ``X-CSRF-Token`` is rejected.

    Documents the double-submit-cookie contract: SameSite=lax catches
    most cross-site form-POSTs, but the explicit header check is
    what closes the corner cases (subdomain cookie injection,
    attacker on a same-registrable-domain origin).
    """
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    _signup(client, email="nocsrf@user.com")
    # Cookie jar now has esther_csrf, but we deliberately don't echo
    # it. A cross-site attacker can't read it via document.cookie, so
    # they couldn't construct this header — the request must be
    # rejected here, not at the route.
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NVDA"},
    )
    assert res.status_code == 403
    assert "CSRF" in res.json()["detail"]


def test_post_with_mismatched_csrf_header_is_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong-value header (attacker guessed) is rejected."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    _signup(client, email="badcsrf@user.com")
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NVDA"},
        headers={"X-CSRF-Token": "not-the-real-token"},
    )
    assert res.status_code == 403


def test_post_with_correct_csrf_header_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: the helper that echoes the cookie produces a 201."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    _signup(client, email="goodcsrf@user.com")
    res = client.post(
        "/api/watchlist/symbols",
        json={"symbol": "NVDA"},
        headers=_csrf(client),
    )
    assert res.status_code == 201


def test_get_does_not_require_csrf_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotent methods are exempt from CSRF enforcement."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL"])
    _signup(client, email="getcsrf@user.com")
    res = client.get("/api/watchlist")
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Per-user snapshot filter — rows scoped to caller's watchlist
# ---------------------------------------------------------------------------


def test_snapshot_filters_rows_to_user_watchlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two users share a controller universe but see only their symbols."""
    monkeypatch.chdir(tmp_path)
    client_a, _ = _build_client(tmp_path, watchlist=["AAPL", "MSFT", "NVDA"])
    _signup(client_a, email="a@user.com")
    _add_symbol(client_a, "AAPL")

    client_b = TestClient(client_a.app)
    _signup(client_b, email="b@user.com")
    _add_symbol(client_b, "MSFT")

    snap_a = client_a.get("/api/snapshot").json()
    snap_b = client_b.get("/api/snapshot").json()

    syms_a = {row["symbol"] for row in snap_a["rows"]}
    syms_b = {row["symbol"] for row in snap_b["rows"]}
    assert syms_a == {"AAPL"}
    assert syms_b == {"MSFT"}


def test_snapshot_filters_ranked_opportunities_to_user_watchlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ranked opportunities are per-symbol too — same filtering applies."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL", "MSFT", "NVDA"])
    _signup(client)
    _add_symbol(client, "AAPL")

    snap = client.get("/api/snapshot").json()
    for opp in snap.get("ranked_opportunities", []):
        assert opp["symbol"] == "AAPL", (
            "ranked_opportunities leaked a symbol outside the user's watchlist"
        )


def test_snapshot_pulse_aggregates_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregate fields (pulse, regime) are watchlist-wide and stay populated."""
    monkeypatch.chdir(tmp_path)
    client, _ = _build_client(tmp_path, watchlist=["AAPL", "MSFT", "NVDA"])
    _signup(client)
    _add_symbol(client, "AAPL")

    snap = client.get("/api/snapshot").json()
    # ``pulse`` describes the operating environment, not the user's
    # positions — filtering would strip useful context. Documented in
    # ``filter_snapshot_for_symbols``.
    assert snap["pulse"] is not None
    assert "tick" in snap
    assert "timestamp" in snap
