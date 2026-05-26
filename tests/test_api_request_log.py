"""Tests for ``RequestLogMiddleware`` and the proxy-trust gating.

Two responsibilities:

1. Every response carries an ``X-Request-ID`` header. The middleware
   honors an inbound id from the client if one is present (frontend
   trace propagation); otherwise it mints a uuid4 hex.
2. ``_client_ip`` only honors ``X-Forwarded-For`` when the app was
   started with ``trust_proxy_headers=True`` — outside that, XFF is
   attacker-controlled and would be a spoofable rate-limit key.

The proxy-trust check goes through the rate-limit dependency end-to-end
rather than calling ``_client_ip`` directly, so the regression covers
the wiring (``app.state.trust_proxy_headers``) in addition to the
function under test.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from src.api import create_app
from src.api.middleware import (
    InMemoryRateLimiter,
    build_rate_limiter_dep,
)
from src.dashboard.controller import MockDashboardController


def _bare_app() -> TestClient:
    """Auth-less app — the simplest path that still wires every middleware."""
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller)
    return TestClient(app)


def test_request_id_header_minted_when_absent() -> None:
    """Every response carries an X-Request-ID header."""
    client = _bare_app()
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    assert "x-request-id" in {k.lower() for k in res.headers}
    # uuid4().hex shape: 32 hex chars.
    assert re.fullmatch(r"[0-9a-f]{32}", res.headers["x-request-id"])


def test_request_id_header_honored_when_supplied() -> None:
    """A client-supplied ``X-Request-ID`` round-trips end-to-end so a
    frontend trace id can correlate across services."""
    client = _bare_app()
    res = client.get(
        "/api/snapshot",
        headers={"X-Request-ID": "trace-abc-123"},
    )
    assert res.status_code == 200
    assert res.headers["x-request-id"] == "trace-abc-123"


def test_request_id_present_even_on_404() -> None:
    """The log middleware is the outermost layer — its header must
    survive on rejected paths too, not just successful 200s."""
    client = _bare_app()
    res = client.get("/api/this-path-does-not-exist")
    assert res.status_code == 404
    assert "x-request-id" in {k.lower() for k in res.headers}


def test_client_ip_ignores_xff_when_proxy_not_trusted() -> None:
    """Default ``trust_proxy_headers=False`` makes the rate limiter
    key on the peer address. Two requests with different XFF values
    but the same peer share the budget — so the spoofing attack
    (rotate XFF to bypass per-IP limit) fails.
    """
    limiter = InMemoryRateLimiter(max_calls=1, per_seconds=60.0)

    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller)
    # Hand-wire one route protected by the limiter so the test stays
    # isolated from /api/auth/* (which has its own limiter wiring).
    dep = build_rate_limiter_dep(limiter, scope="t")
    from fastapi import Depends

    @app.get("/__rl_test")
    async def _rl(_: None = Depends(dep)) -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)

    # First call from any spoofed XFF: passes (consumes the budget).
    r1 = client.get("/__rl_test", headers={"X-Forwarded-For": "1.1.1.1"})
    assert r1.status_code == 200
    # Second call from a different spoofed XFF, same peer: rate-limited
    # because the limiter ignored the spoof and keyed on peer instead.
    r2 = client.get("/__rl_test", headers={"X-Forwarded-For": "2.2.2.2"})
    assert r2.status_code == 429


def test_client_ip_honors_xff_when_proxy_trusted() -> None:
    """Flip ``trust_proxy_headers=True`` and the per-IP budget is
    keyed on the first-hop XFF — which is correct iff the operator
    has actually placed a trusted proxy in front of the server.
    """
    limiter = InMemoryRateLimiter(max_calls=1, per_seconds=60.0)

    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(controller)
    # Flip the trust flag after the app is built — the middleware
    # reads it off ``app.state`` on every request, so post-hoc flips
    # are observable. (Production sets it once at startup.)
    app.state.trust_proxy_headers = True

    dep = build_rate_limiter_dep(limiter, scope="t")
    from fastapi import Depends

    @app.get("/__rl_test_trusted")
    async def _rl(_: None = Depends(dep)) -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)

    # Each request asserts a fresh XFF identity, so each has its own
    # budget and both pass.
    r1 = client.get("/__rl_test_trusted", headers={"X-Forwarded-For": "1.1.1.1"})
    r2 = client.get("/__rl_test_trusted", headers={"X-Forwarded-For": "2.2.2.2"})
    assert r1.status_code == 200
    assert r2.status_code == 200
