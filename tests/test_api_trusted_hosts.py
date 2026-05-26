"""Tests for ``TrustedHostMiddleware`` wiring.

When ``Settings.trusted_hosts`` is anything other than the default
``["*"]``, Starlette's :class:`TrustedHostMiddleware` is wired and
mismatched ``Host`` headers are rejected with 400 before any other
middleware runs. Host-header injection (cache poisoning, password-
reset link forgery, virtual-host confusion) is otherwise unmitigated.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController


def test_no_trusted_hosts_means_wildcard_pass() -> None:
    """Default ``["*"]`` accepts any Host — keeps the dev workflow
    (where the test client and Vite proxy both forge their own Host)
    working out of the box."""
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller)
    client = TestClient(app)
    res = client.get(
        "/api/health", headers={"Host": "literally-anything.example"}
    )
    assert res.status_code == 200


def test_trusted_hosts_allow_matching_host() -> None:
    """A request whose Host matches the configured allow-list passes
    straight through."""
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, trusted_hosts=["esther.example.com"])
    client = TestClient(app, base_url="http://esther.example.com")
    res = client.get("/api/health")
    assert res.status_code == 200


def test_trusted_hosts_reject_mismatched_host() -> None:
    """A spoofed Host is rejected with 400. The route handler never
    runs — the response comes from TrustedHostMiddleware itself."""
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller, trusted_hosts=["esther.example.com"])
    client = TestClient(app, base_url="http://attacker.example.com")
    res = client.get("/api/health")
    assert res.status_code == 400
