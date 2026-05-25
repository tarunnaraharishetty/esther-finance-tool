"""User accounts + session API surface.

Four routes, all backed by :class:`UserStore`:

* ``POST /api/auth/signup`` — create a new account, return user +
  set session cookie.
* ``POST /api/auth/login``  — verify credentials, set session cookie.
* ``POST /api/auth/logout`` — revoke current session, clear cookie.
* ``GET  /api/auth/me``     — return the current user or ``null``.

Session model
-------------
Login mints a random URL-safe token, persists it in the ``sessions``
table with a TTL, and ships it back as a signed HTTP-only cookie.
Every subsequent request can be resolved to a user via
:func:`get_current_user` — the dependency reads the cookie, verifies
the signature, and looks up the token in the store. A stolen cookie
is useless once :meth:`UserStore.revoke_session` is called.

Why not JWT
-----------
JWT trades revocation flexibility for stateless validation. At a
single-process scale that's a bad trade — adding a revocation list
duplicates the work a session table already does. Session cookies
let us delete one row and invalidate any in-flight token.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Cookie, Depends, FastAPI, HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, Field, field_validator

from src.data.user_store import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    User,
    UserStore,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


# Salt for the itsdangerous serializer. Cookie payload =
# ``serializer.dumps(session_token)``; the serializer signs with
# the app's secret + this salt. Two different salts let us reuse
# one secret for different cookie families without cross-collision.
_COOKIE_SALT = "esther-session-v1"


class SignupRequest(BaseModel):
    """Request body for ``POST /api/auth/signup``."""

    email: str = Field(min_length=3, max_length=320)
    # Bare minimum: non-empty, at least 8 chars. We deliberately
    # don't enforce complexity rules — modern password guidance
    # (NIST 800-63B) favors length over composition.
    password: str = Field(min_length=8, max_length=200)

    @field_validator("email")
    @classmethod
    def _email_has_at(cls, v: str) -> str:
        """Cheap email shape check. We deliberately don't pull in
        ``email-validator`` for v1 — the UserStore stores anything
        that round-trips through the unique index, and the bigger
        risk is "wrong email" not "malformed email"."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("email must contain '@' and a domain")
        return v


class LoginRequest(BaseModel):
    """Request body for ``POST /api/auth/login``."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=200)


def register_auth_routes(
    app: FastAPI,
    *,
    user_store: UserStore,
    session_secret: str,
    session_ttl_days: int,
    cookie_name: str,
    secure_cookies: bool = False,
    post_signup_hook: Callable[[User], None] | None = None,
    login_rate_limit_dep: Callable[..., Awaitable[None]] | None = None,
    signup_rate_limit_dep: Callable[..., Awaitable[None]] | None = None,
) -> None:
    """Attach ``/api/auth/*`` routes to ``app``.

    Args:
        app: FastAPI app to extend.
        user_store: Bound :class:`UserStore`. Route closes over it
            so the same instance backs every call.
        session_secret: Cookie-signing key. In production this comes
            from ``Settings.session_secret_key`` (an env-loaded
            ``SecretStr``); the dev default is a placeholder string.
        session_ttl_days: Session lifetime in days. Persisted as the
            ``expires_at`` column AND used to set the cookie max-age.
        cookie_name: Cookie name to issue. Stable across deployments.
        secure_cookies: When True, marks cookies ``Secure`` so they
            only ship over HTTPS. Off in dev (localhost is HTTP);
            production deployments should pass True.
        post_signup_hook: Optional callback invoked with the new
            :class:`User` after a successful signup, before the
            response is returned. ``app.py`` wires the watchlist
            default-seed through this hook so first-run users land
            on a populated dashboard. **Hook exceptions abort signup
            with a 503 response** (B-14): the hook owns its own
            rollback (purging partially-seeded rows and deleting the
            user) so signup is atomic from the client's perspective.
            The previous best-effort behavior produced an account
            whose first dashboard view was empty — users reasonably
            concluded auth was broken.
        login_rate_limit_dep: Optional FastAPI dependency applied to
            ``/login`` only. ``app.py`` wires a tighter per-IP budget
            here (10 attempts / 5 min) to bound brute-force +
            credential-stuffing. ``None`` disables limiting on
            ``/login`` (the default; tests and local dev).
        signup_rate_limit_dep: Optional FastAPI dependency applied to
            ``/signup`` only. ``app.py`` wires a separate per-IP budget
            here (3 attempts / hour) — signup is rarer than login and
            the abuse pattern is account-creation flooding, which
            warrants a strict longer-window cap independent of the
            login budget. ``None`` disables limiting on ``/signup``.
    """
    serializer = URLSafeSerializer(session_secret, salt=_COOKIE_SALT)
    ttl = timedelta(days=session_ttl_days)
    ttl_seconds = int(ttl.total_seconds())

    # Stash on app.state so any other route (and tests) can resolve
    # the current user without a separate import path.
    app.state.user_store = user_store
    app.state.session_serializer = serializer
    app.state.session_cookie_name = cookie_name

    router = APIRouter(prefix="/api/auth", tags=["auth"])
    signup_dependencies = (
        [Depends(signup_rate_limit_dep)] if signup_rate_limit_dep else []
    )
    login_dependencies = (
        [Depends(login_rate_limit_dep)] if login_rate_limit_dep else []
    )

    @router.post("/signup", status_code=201, dependencies=signup_dependencies)
    async def signup(body: SignupRequest, response: Response) -> dict[str, Any]:
        """Create a new account + log the user in.

        Returns 201 with ``{user}`` and sets the session cookie on
        the response. 409 if the email is already registered.
        """
        try:
            user = user_store.create_user(body.email, body.password)
        except EmailAlreadyRegistered:
            raise HTTPException(
                status_code=409,
                detail="An account with that email already exists.",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        # Post-signup hook (e.g. watchlist default seed). Hook
        # failures abort signup with 503 — the hook is responsible
        # for its own rollback so by the time we re-raise, the user
        # row no longer exists. The trader gets a clean retry instead
        # of an account whose first dashboard view is empty (B-14).
        if post_signup_hook is not None:
            try:
                post_signup_hook(user)
            except Exception as exc:
                log.warning(
                    "auth.signup.post_hook_failed",
                    user_id=user.id,
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Could not finish setting up the account. "
                        "Please try again in a moment."
                    ),
                ) from None

        session = user_store.create_session(user.id, ttl)
        _set_session_cookie(
            response,
            serializer.dumps(session.token),
            cookie_name=cookie_name,
            ttl_seconds=ttl_seconds,
            secure=secure_cookies,
        )
        log.info("auth.signup.ok", user_id=user.id, email=user.email)
        return {"user": user.to_wire()}

    @router.post("/login", dependencies=login_dependencies)
    async def login(body: LoginRequest, response: Response) -> dict[str, Any]:
        """Verify credentials + mint a session.

        Returns ``{user}`` on success, 401 on bad credentials. The
        error message is intentionally generic so the response
        doesn't leak which half (email vs. password) was wrong.
        """
        try:
            user = user_store.verify_login(body.email, body.password)
        except InvalidCredentials:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password.",
            ) from None

        session = user_store.create_session(user.id, ttl)
        _set_session_cookie(
            response,
            serializer.dumps(session.token),
            cookie_name=cookie_name,
            ttl_seconds=ttl_seconds,
            secure=secure_cookies,
        )
        log.info("auth.login.ok", user_id=user.id, email=user.email)
        return {"user": user.to_wire()}

    @router.post("/logout")
    async def logout(
        response: Response,
        session_cookie: str | None = Cookie(default=None, alias=cookie_name),
    ) -> dict[str, Any]:
        """Revoke the current session + clear the cookie.

        Always returns 200 — logging out an already-logged-out user
        is a no-op, not an error.
        """
        if session_cookie:
            token = _decode_cookie(serializer, session_cookie)
            if token:
                revoked = user_store.revoke_session(token)
                log.info("auth.logout.revoked", revoked=revoked)
        # Always clear the cookie, even if there was nothing to revoke.
        _clear_session_cookie(
            response, cookie_name=cookie_name, secure=secure_cookies
        )
        return {"ok": True}

    @router.get("/me")
    async def me(request: Request) -> dict[str, Any]:
        """Return the current user or ``{user: null}`` for guests.

        Never 401s — anonymous is a valid state and the frontend
        keys off ``user === null`` to render the login screen.
        """
        user = await get_current_user(request)
        return {"user": user.to_wire() if user is not None else None}

    app.include_router(router)


# ---------------------------------------------------------------------------
# Request-time session resolver
# ---------------------------------------------------------------------------


async def get_current_user(request: Request) -> User | None:
    """Resolve the session cookie on ``request`` to a User.

    Designed as a FastAPI dependency. Returns ``None`` when:
    - the cookie is missing
    - the signature is bad (cookie was tampered with)
    - the underlying session is expired or revoked
    - the user store isn't wired on this app

    Routes that *require* a user should depend on
    :func:`require_current_user` instead, which raises 401 on None.
    """
    user_store: UserStore | None = getattr(request.app.state, "user_store", None)
    serializer: URLSafeSerializer | None = getattr(
        request.app.state, "session_serializer", None
    )
    cookie_name: str | None = getattr(
        request.app.state, "session_cookie_name", None
    )
    if user_store is None or serializer is None or cookie_name is None:
        return None
    raw_cookie = request.cookies.get(cookie_name)
    if not raw_cookie:
        return None
    token = _decode_cookie(serializer, raw_cookie)
    if token is None:
        return None
    return user_store.lookup_session(token)


async def require_current_user(request: Request) -> User:
    """Same as :func:`get_current_user` but raises 401 instead of
    returning None. Use as a FastAPI dependency on routes that
    must reject anonymous traffic."""
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------


def _set_session_cookie(
    response: Response,
    signed_value: str,
    *,
    cookie_name: str,
    ttl_seconds: int,
    secure: bool,
) -> None:
    """Issue a fresh session cookie. HttpOnly + SameSite=lax by default."""
    response.set_cookie(
        key=cookie_name,
        value=signed_value,
        max_age=ttl_seconds,
        httponly=True,
        secure=secure,
        samesite="lax",
        # Default to root path so every API route sees it; in a
        # subpath deployment the operator overrides via reverse proxy.
        path="/",
    )


def _clear_session_cookie(
    response: Response, *, cookie_name: str, secure: bool
) -> None:
    """Empty the cookie + zero the max-age so the browser drops it."""
    response.delete_cookie(
        key=cookie_name,
        path="/",
        httponly=True,
        secure=secure,
        samesite="lax",
    )


def _decode_cookie(serializer: URLSafeSerializer, value: str) -> str | None:
    """Unwrap a signed cookie back into the session token, or None
    if the signature doesn't verify."""
    try:
        decoded = serializer.loads(value)
    except BadSignature:
        return None
    if not isinstance(decoded, str):
        return None
    return decoded


__all__ = [
    "get_current_user",
    "register_auth_routes",
    "require_current_user",
]
