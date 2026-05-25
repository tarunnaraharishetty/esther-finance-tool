"""ASGI middleware: auth gate + CSRF gate + per-IP rate limiter.

Four small helpers that wrap the FastAPI app without forcing every
route module to be aware of them:

* :func:`AuthGateMiddleware` — when a UserStore is wired on the app,
  every ``/api/*`` request must carry a valid session cookie.
  Whitelist: ``/api/health`` (load-balancer probes), ``/api/auth/*``
  (login / signup / me / logout themselves), and any ``OPTIONS``
  preflight (browser CORS — the response carries the auth cookie on
  the follow-up real request, not on the preflight).

* :class:`CSRFMiddleware` — double-submit cookie defense against
  cross-site state-changing requests. State-changing methods
  (POST/PUT/PATCH/DELETE) on ``/api/*`` (except ``/api/auth/*`` +
  ``/api/health``) must echo the ``esther_csrf`` cookie back in an
  ``X-CSRF-Token`` header. SameSite=lax already blocks the most
  common cross-site form-POST, but it doesn't help against
  subdomain-cookie injection or attacker-controlled origins under
  the same registrable domain — the explicit header check does.

* :class:`InMemoryRateLimiter` + :func:`build_rate_limiter_dep` — a
  sliding-window per-IP-per-route limiter backed by a deque. Cheap
  and dependency-free; suitable for single-process deployments. Use
  it as a FastAPI dependency on individual routes that cost money to
  serve (login, signup).

* :class:`PathRateLimitMiddleware` — applies an
  :class:`InMemoryRateLimiter` to a configured set of path prefixes
  without modifying each route module. Use this for the LLM-backed
  endpoints (research, analyzer LLM mode, compare narrative) where
  threading a dependency through every register function would be
  louder than the protection is worth.

All pieces stay deliberately small. A multi-process or
multi-instance deployment will want to swap the in-memory limiter
for a Redis backend; the limiter is the seam for that.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from threading import Lock

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.api.auth import CSRF_COOKIE_NAME, CSRF_HEADER_NAME, get_current_user
from src.utils.logging import get_logger

log = get_logger(__name__)


# Routes that must remain reachable without auth, no matter what.
# - ``/api/health``: liveness probe — load balancers and uptime
#   monitors hit this anonymously.
# - ``/api/readyz``: readiness probe — same operator-tool callers as
#   /api/health, never carries cookies.
# - ``/api/auth/*``: login / signup / me / logout endpoints themselves.
_PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/api/health",
    "/api/readyz",
    "/api/auth/",
)


class AuthGateMiddleware(BaseHTTPMiddleware):
    """Reject anonymous traffic on ``/api/*`` when auth is configured.

    "Configured" means ``app.state.user_store`` is non-None — i.e. the
    UserStore was wired during ``create_app``. When auth is disabled
    (the single-user dev workflow that pre-dates accounts) the
    middleware is a no-op pass-through.

    The middleware does **not** filter responses or attach a user; it
    only gates the request. Routes that need the user object continue
    to inject :func:`get_current_user` / :func:`require_current_user`.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        method = request.method.upper()
        if not path.startswith("/api/"):
            return await call_next(request)
        if method == "OPTIONS":
            # CORS preflight. The browser never carries the session
            # cookie on the preflight — only on the actual request.
            return await call_next(request)
        if any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES):
            return await call_next(request)
        if getattr(request.app.state, "user_store", None) is None:
            # Auth not wired on this app — pass through.
            return await call_next(request)
        user = await get_current_user(request)
        if user is None:
            return JSONResponse(
                {"detail": "Authentication required."},
                status_code=401,
            )
        # Cache the resolved user on the request so downstream
        # middleware (PathRateLimit) doesn't have to re-look-up the
        # session for the same request. The route-level dependency
        # ``get_current_user`` still gets called by FastAPI for
        # handlers that declare it — that's a separate code path and
        # exempt from this cache.
        request.state.current_user = user
        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class InMemoryRateLimiter:
    """Per-key sliding-window counter.

    ``check(key)`` returns True iff the call is within budget for the
    current window; it also records the call. The window is a deque
    of recent timestamps trimmed on every check, so memory is bounded
    by the number of recent active keys × max_calls.

    Thread-safe via a single lock — fine at the request rates this
    process serves; a Redis-backed implementation is the right
    upgrade for multi-instance deployments.
    """

    def __init__(self, *, max_calls: int, per_seconds: float) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if per_seconds <= 0:
            raise ValueError("per_seconds must be > 0")
        self._max_calls = max_calls
        self._window = per_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> bool:
        """Record one hit for ``key`` and return whether it fit in budget."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            dq = self._hits[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max_calls:
                return False
            dq.append(now)
            return True

    def reset(self) -> None:
        """Clear all tracked keys. Intended for tests."""
        with self._lock:
            self._hits.clear()


def _client_ip(request: Request) -> str:
    """Best-effort client identifier for rate-limit keying.

    Order of preference:
    - ``X-Forwarded-For`` first hop (when behind a trusted reverse proxy);
    - ``request.client.host`` (uvicorn-resolved peer address);
    - ``unknown`` fallback so a missing peer doesn't crash the limiter.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _rate_limit_subject(request: Request) -> str:
    """Pick a stable rate-limit key for ``request``.

    Authenticated requests key on the resolved user id so an attacker
    can't rotate IPs (Tor, residential proxies) to bypass a per-user
    cap on an expensive route. Anonymous requests fall back to the
    client IP, which is the strongest identifier we have. The user is
    expected to have been resolved + cached on ``request.state`` by
    :class:`AuthGateMiddleware`; for routes that bypass the gate
    (``/api/health``, ``/api/auth/*``) this resolves to IP.
    """
    user = getattr(request.state, "current_user", None)
    if user is not None:
        return f"user:{user.id}"
    return f"ip:{_client_ip(request)}"


def build_rate_limiter_dep(
    limiter: InMemoryRateLimiter, *, scope: str
) -> Callable[[Request], Awaitable[None]]:
    """Build a FastAPI dependency that enforces ``limiter`` under ``scope``.

    Scope partitions the limiter so unrelated routes don't share a
    budget — e.g. login attempts and signup attempts are tracked
    separately even though both originate from the same IP.

    Raises 429 on overflow, with ``Retry-After`` set conservatively to
    one window length so well-behaved clients back off.
    """
    window_seconds = limiter._window

    async def _dep(request: Request) -> None:
        key = f"{scope}:{_client_ip(request)}"
        if not limiter.check(key):
            log.warning(
                "ratelimit.exceeded",
                scope=scope,
                ip=_client_ip(request),
                path=request.url.path,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Slow down and try again.",
                headers={"Retry-After": str(int(window_seconds))},
            )

    return _dep


def install_auth_gate(app: ASGIApp) -> None:
    """Attach :class:`AuthGateMiddleware` to ``app``.

    Kept as a small helper so ``create_app`` can wire the middleware
    in one line and tests can verify the middleware is present.
    """
    # Starlette's ``add_middleware`` is the documented public API.
    # Cast is fine here — ``app`` is a FastAPI instance in practice.
    app.add_middleware(AuthGateMiddleware)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CSRF (double-submit cookie)
# ---------------------------------------------------------------------------


# Methods that mutate server state and therefore must carry a CSRF
# token. Read-only methods (GET, HEAD) are exempt; OPTIONS is handled
# upstream by CORS preflight.
_CSRF_STATE_CHANGING_METHODS: frozenset[str] = frozenset(
    {"POST", "PUT", "PATCH", "DELETE"}
)

# Path prefixes exempt from CSRF enforcement. ``/api/auth/*`` is the
# bootstrap surface — a logged-out client has no CSRF cookie yet.
# ``/api/health`` and ``/api/readyz`` are probe endpoints that never
# mutate state. The CSRF middleware also short-circuits on GET/HEAD
# methods, so these prefixes are belt-and-braces against a future
# POST to a health route.
_CSRF_EXEMPT_PREFIXES: tuple[str, ...] = (
    "/api/auth/",
    "/api/health",
    "/api/readyz",
)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Enforce double-submit-cookie CSRF on state-changing API calls.

    Activation is opt-in: when ``app.state.csrf_enabled`` is False
    (the default for the no-auth dev workflow) the middleware is a
    pass-through. ``create_app`` flips it on when a UserStore is
    wired.

    Verification: read the ``esther_csrf`` cookie + the
    ``X-CSRF-Token`` header, compare with ``secrets.compare_digest``
    (constant-time so a timing side-channel can't leak the token).
    Reject 403 on missing or mismatched.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not getattr(request.app.state, "csrf_enabled", False):
            return await call_next(request)
        path = request.url.path
        method = request.method.upper()
        if method not in _CSRF_STATE_CHANGING_METHODS:
            return await call_next(request)
        if not path.startswith("/api/"):
            return await call_next(request)
        if any(path.startswith(prefix) for prefix in _CSRF_EXEMPT_PREFIXES):
            return await call_next(request)

        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        header_token = request.headers.get(CSRF_HEADER_NAME)
        if not cookie_token or not header_token:
            log.warning(
                "csrf.reject.missing",
                path=path,
                has_cookie=bool(cookie_token),
                has_header=bool(header_token),
            )
            return JSONResponse(
                {"detail": "Missing CSRF token."},
                status_code=403,
            )
        if not secrets.compare_digest(cookie_token, header_token):
            log.warning("csrf.reject.mismatch", path=path)
            return JSONResponse(
                {"detail": "Invalid CSRF token."},
                status_code=403,
            )
        return await call_next(request)


def install_csrf_gate(app: ASGIApp) -> None:
    """Attach :class:`CSRFMiddleware` to ``app``.

    Mirrors :func:`install_auth_gate` so ``create_app`` can wire the
    two gates next to each other; the actual enforcement is gated on
    ``app.state.csrf_enabled`` so the no-auth dev workflow stays
    untouched.
    """
    app.add_middleware(CSRFMiddleware)  # type: ignore[attr-defined]


class PathRateLimitMiddleware(BaseHTTPMiddleware):
    """Throttle requests whose path matches one of the configured prefixes.

    Configuration is an iterable of ``(prefix, scope, limiter)``
    tuples. The first matching prefix wins. Keys are
    ``"{scope}:{ip}"`` so unrelated routes don't share budget.

    Kept separate from :class:`AuthGateMiddleware` so the dev-time
    no-auth workflow can still pick up rate limiting on the
    LLM-backed routes.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        rules: list[tuple[str, str, InMemoryRateLimiter]],
    ) -> None:
        super().__init__(app)
        self._rules = list(rules)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        method = request.method.upper()
        if method == "OPTIONS":
            return await call_next(request)
        for prefix, scope, limiter in self._rules:
            if path.startswith(prefix):
                key = f"{scope}:{_rate_limit_subject(request)}"
                if not limiter.check(key):
                    log.warning(
                        "ratelimit.exceeded",
                        scope=scope,
                        ip=_client_ip(request),
                        path=path,
                    )
                    retry_after = int(limiter._window)
                    return JSONResponse(
                        {
                            "detail": (
                                "Rate limit exceeded for this endpoint. "
                                "Slow down and try again."
                            )
                        },
                        status_code=429,
                        headers={"Retry-After": str(retry_after)},
                    )
                break
        return await call_next(request)


def install_path_rate_limits(
    app: ASGIApp, rules: list[tuple[str, str, InMemoryRateLimiter]]
) -> None:
    """Attach :class:`PathRateLimitMiddleware` with ``rules``.

    No-op when ``rules`` is empty so callers can pass a conditional
    list without branching.
    """
    if not rules:
        return
    app.add_middleware(PathRateLimitMiddleware, rules=rules)  # type: ignore[attr-defined]


__all__ = [
    "AuthGateMiddleware",
    "CSRFMiddleware",
    "InMemoryRateLimiter",
    "PathRateLimitMiddleware",
    "build_rate_limiter_dep",
    "install_auth_gate",
    "install_csrf_gate",
    "install_path_rate_limits",
]
