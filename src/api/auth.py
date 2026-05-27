"""User accounts + session API surface.

Ten routes, all backed by :class:`UserStore`:

* ``POST /api/auth/signup`` — create a new account, return user +
  set session cookie. Auto-dispatches a verification email (best
  effort — signup succeeds even if SMTP is unreachable).
* ``POST /api/auth/login``  — verify credentials, set session cookie.
* ``POST /api/auth/logout`` — revoke current session, clear cookie.
* ``GET  /api/auth/me``     — return the current user or ``null``.
* ``POST /api/auth/password-reset/request`` — mint a reset token,
  email it. Always returns 200 (no email enumeration).
* ``POST /api/auth/password-reset/confirm`` — redeem a token + set
  a new password + revoke every session for the user.
* ``POST /api/auth/verify/request`` — authenticated. Re-mint + re-
  send a verification email to the current user. No-op (200) when
  the user is already verified.
* ``POST /api/auth/verify/confirm`` — public, redeem a verification
  token. Flips ``email_verified=True`` on the user.
* ``POST /api/auth/password/change`` — authenticated. Verify the
  current password, set a new one, revoke every other session for
  the user (rotate the current one so the caller stays logged in).
* ``DELETE /api/auth/account`` — authenticated. Re-auth via
  password, hard-delete the user + cross-store data.

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

import secrets
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel, Field, field_validator

from src.api.email_verification_emailer import (
    EmailVerificationEmailer,
    LogEmailVerificationEmailer,
)
from src.api.password_reset_emailer import (
    LogPasswordResetEmailer,
    PasswordResetEmailer,
)
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

# CSRF double-submit cookie + header names. Lives here (not in
# middleware.py) so cookie helpers + middleware pull from one source
# without a circular import — middleware already depends on this
# module for ``get_current_user``. Cookie is intentionally NOT
# HttpOnly so the frontend can read it via ``document.cookie`` and
# echo it as ``X-CSRF-Token``; a cross-origin attacker can't read it
# because ``document.cookie`` is same-origin scoped.
CSRF_COOKIE_NAME = "esther_csrf"
CSRF_HEADER_NAME = "x-csrf-token"


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


class PasswordResetRequestBody(BaseModel):
    """Body for ``POST /api/auth/password-reset/request``."""

    email: str = Field(min_length=3, max_length=320)


class PasswordResetConfirmBody(BaseModel):
    """Body for ``POST /api/auth/password-reset/confirm``."""

    # url-safe base64 of 32 random bytes from the store side. 80 is a
    # generous upper bound — actual length is 43.
    token: str = Field(min_length=20, max_length=80)
    # Same length floor as ``SignupRequest.password``.
    new_password: str = Field(min_length=8, max_length=200)


class EmailVerificationConfirmBody(BaseModel):
    """Body for ``POST /api/auth/verify/confirm``."""

    token: str = Field(min_length=20, max_length=80)


class AccountDeleteBody(BaseModel):
    """Body for ``DELETE /api/auth/account``. Password re-entry
    bounds the blast radius of a stolen session cookie."""

    password: str = Field(min_length=1, max_length=200)


class PasswordChangeBody(BaseModel):
    """Body for ``POST /api/auth/password/change``.

    The current password is required for re-auth so a hijacked
    session cookie can't silently rotate the credential. Minimums
    match :class:`SignupRequest`.
    """

    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=8, max_length=200)


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
    password_reset_emailer: PasswordResetEmailer | None = None,
    password_reset_ttl: timedelta = timedelta(hours=1),
    password_reset_base_url: str | None = None,
    password_reset_request_rate_limit_dep: Callable[..., Awaitable[None]] | None = None,
    password_reset_confirm_rate_limit_dep: Callable[..., Awaitable[None]] | None = None,
    email_verification_emailer: EmailVerificationEmailer | None = None,
    email_verification_ttl: timedelta = timedelta(hours=24),
    email_verification_base_url: str | None = None,
    email_verification_request_rate_limit_dep: Callable[..., Awaitable[None]] | None = None,
    on_account_deleted: Callable[[int], None] | None = None,
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

    # Emailer defaults to the log transport — operator running a
    # closed beta gets the link in the structured log and can relay
    # manually. ``app.py`` wires SmtpPasswordResetEmailer when
    # SMTP_HOST is set.
    emailer: PasswordResetEmailer = (
        password_reset_emailer
        if password_reset_emailer is not None
        else LogPasswordResetEmailer()
    )
    verify_emailer: EmailVerificationEmailer = (
        email_verification_emailer
        if email_verification_emailer is not None
        else LogEmailVerificationEmailer()
    )

    def _build_verify_url(request: Request, token: str) -> str:
        """Build the verification link the email points at.

        Same shape as the password-reset URL: a single page on the
        frontend (``/verify-email?token=…``) that POSTs the token
        back via the confirm route. ``email_verification_base_url``
        overrides the auto-derived origin when API + frontend live
        on different hosts.
        """
        base = (
            email_verification_base_url
            if email_verification_base_url
            else f"{request.url.scheme}://{request.url.netloc}"
        )
        return f"{base.rstrip('/')}/verify-email?token={quote(token)}"

    async def _send_verification_for(user: User, request: Request) -> None:
        """Mint a token + dispatch the link. Best-effort: any
        transport failure logs but doesn't raise — signup must
        succeed even when SMTP is down (the user can re-request
        verification later via /verify/request)."""
        if user.email_verified:
            return
        token = user_store.create_email_verification_token(
            user.id, email_verification_ttl
        )
        verify_url = _build_verify_url(request, token)
        await run_in_threadpool(verify_emailer.send, user.email, verify_url)
        log.info(
            "auth.email_verification.dispatched",
            user_id=user.id,
            email=user.email,
        )

    router = APIRouter(prefix="/api/auth", tags=["auth"])
    signup_dependencies = (
        [Depends(signup_rate_limit_dep)] if signup_rate_limit_dep else []
    )
    login_dependencies = (
        [Depends(login_rate_limit_dep)] if login_rate_limit_dep else []
    )
    pwreset_request_dependencies = (
        [Depends(password_reset_request_rate_limit_dep)]
        if password_reset_request_rate_limit_dep
        else []
    )
    pwreset_confirm_dependencies = (
        [Depends(password_reset_confirm_rate_limit_dep)]
        if password_reset_confirm_rate_limit_dep
        else []
    )
    verify_request_dependencies = (
        [Depends(email_verification_request_rate_limit_dep)]
        if email_verification_request_rate_limit_dep
        else []
    )

    @router.post("/signup", status_code=201, dependencies=signup_dependencies)
    async def signup(
        body: SignupRequest, request: Request, response: Response
    ) -> dict[str, Any]:
        """Create a new account + log the user in.

        Returns 201 with ``{user}`` and sets the session cookie on
        the response. 409 if the email is already registered.

        Side effect: dispatches a verification email to the new
        address. Best-effort — SMTP transport errors are swallowed
        at the emailer (the user can re-request via /verify/request
        if the email doesn't arrive). The verification banner stays
        up on the frontend until the user redeems the token.
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
        _set_csrf_cookie_on(response, secure=secure_cookies, ttl_seconds=ttl_seconds)
        # Auto-dispatch verification email. Failures inside the
        # emailer are already swallowed; wrap the mint+dispatch in
        # one more try/except so a UserStore error (extremely
        # unlikely — the user row was just created) doesn't abort
        # the signup response either. The frontend banner + a
        # /verify/request retry are the recovery path.
        try:
            await _send_verification_for(user, request)
        except Exception as exc:
            log.warning(
                "auth.signup.verify_dispatch_failed",
                user_id=user.id,
                error=str(exc),
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
        _set_csrf_cookie_on(response, secure=secure_cookies, ttl_seconds=ttl_seconds)
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
        # Always clear both cookies, even if there was nothing to
        # revoke server-side — a stale CSRF cookie would otherwise
        # outlive the session and confuse the next login.
        _clear_session_cookie(
            response, cookie_name=cookie_name, secure=secure_cookies
        )
        _clear_csrf_cookie_on(response, secure=secure_cookies)
        return {"ok": True}

    @router.post(
        "/password-reset/request",
        status_code=200,
        dependencies=pwreset_request_dependencies,
    )
    async def password_reset_request(
        body: PasswordResetRequestBody, request: Request
    ) -> dict[str, Any]:
        """Mint a single-use reset token + email it.

        Security contract:

        * **Always returns 200 with the same message**, whether or not
          the email belongs to a registered account. Any other shape
          (404, "we couldn't find that email", different latency)
          becomes a registered-account enumeration oracle.
        * The emailer ``send`` call swallows transport errors so an
          SMTP outage can't diverge the response either.
        * Rate-limited per IP via the dependency so a script can't
          flood the SMTP relay (or the log sink for the default
          transport).

        Token TTL is 1h by default — long enough for users to read
        mail, short enough that an intercepted link isn't a long-
        lived credential.
        """
        # Look up the user — silently no-op when missing so we never
        # leak account presence via the response shape.
        user = user_store.get_user_by_email(body.email)
        if user is not None:
            token = user_store.create_password_reset_token(
                user.id, password_reset_ttl
            )
            base = (
                password_reset_base_url
                if password_reset_base_url
                else f"{request.url.scheme}://{request.url.netloc}"
            )
            reset_url = f"{base.rstrip('/')}/reset-password?token={quote(token)}"
            # Run the (potentially blocking) email send off-thread so
            # the request loop stays responsive.
            await run_in_threadpool(emailer.send, user.email, reset_url)
            log.info(
                "auth.password_reset.requested",
                user_id=user.id,
                email=user.email,
            )
        else:
            # No such email. Log at INFO with no PII beyond the
            # submitted value so abuse patterns are observable.
            log.info(
                "auth.password_reset.unknown_email",
                email=body.email,
            )
        # Identical message on both branches — uniform timing relies
        # on Python's normal control flow being unaffected by the
        # branch (the UserStore lookup runs in both cases; only the
        # token mint + email send differ, and both happen off the
        # response path latency-wise).
        return {
            "ok": True,
            "message": (
                "If that email is registered, a reset link is on its way."
            ),
        }

    @router.post(
        "/password-reset/confirm",
        status_code=200,
        dependencies=pwreset_confirm_dependencies,
    )
    async def password_reset_confirm(
        body: PasswordResetConfirmBody, response: Response
    ) -> dict[str, Any]:
        """Redeem a token + set the new password.

        Three-step sequence the UserStore exposes as separate methods:

        1. ``consume_password_reset_token`` — atomic mark-used; returns
           the bound user if the token is valid (exists, unused, not
           yet expired).
        2. ``update_password`` — re-hashes via bcrypt at the same
           cost factor as create_user.
        3. ``revoke_all_sessions_for_user`` — defense in depth: any
           old session cookie an attacker held becomes useless the
           moment the password changes.

        Also clears the caller's own session + CSRF cookies in case
        they hit this endpoint while still holding a stale session.
        The user has to log in again with their new password.

        Bad token / expired / already-used all collapse into one 400
        with a generic message — surfacing the exact failure mode
        would let an attacker enumerate the token state machine.
        """
        user = user_store.consume_password_reset_token(body.token)
        if user is None:
            log.warning("auth.password_reset.invalid_token")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Reset link is invalid or has expired. "
                    "Request a new one."
                ),
            )
        if not user_store.update_password(user.id, body.new_password):
            # Defensive — consume succeeded but update missed. Should
            # be impossible given the user row's existence is what
            # made the FK on the token valid in the first place.
            log.error(
                "auth.password_reset.update_failed",
                user_id=user.id,
            )
            raise HTTPException(
                status_code=500,
                detail="Could not update password. Please try again.",
            )
        revoked = user_store.revoke_all_sessions_for_user(user.id)
        log.info(
            "auth.password_reset.confirmed",
            user_id=user.id,
            sessions_revoked=revoked,
        )
        # Belt-and-braces: clear cookies on the response too.
        _clear_session_cookie(
            response, cookie_name=cookie_name, secure=secure_cookies
        )
        _clear_csrf_cookie_on(response, secure=secure_cookies)
        return {"ok": True}

    @router.post(
        "/verify/request",
        status_code=200,
        dependencies=verify_request_dependencies,
    )
    async def verify_request(request: Request) -> dict[str, Any]:
        """Re-send the verification email to the current user.

        Authenticated route — relies on the auth gate having
        resolved a user before this handler runs. Returns 401 if
        unauthenticated. No-op (200) when the user is already
        verified — same wire shape so the frontend banner can call
        unconditionally during reconciliation.

        Rate-limited to bound the email queue (real or operator-
        manual) under accidental re-click spam.
        """
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if user.email_verified:
            return {"ok": True, "already_verified": True}
        await _send_verification_for(user, request)
        return {"ok": True, "already_verified": False}

    @router.post("/verify/confirm", status_code=200)
    async def verify_confirm(
        body: EmailVerificationConfirmBody,
    ) -> dict[str, Any]:
        """Public route — redeem a verification token.

        Same shape as the password-reset confirm: invalid / expired /
        replayed all collapse to one generic 400. The token's
        consume path atomically marks it used + flips the user's
        ``email_verified`` bit (one transaction over two UPDATEs).

        Idempotent on already-verified users — the consume path is
        a no-op flip when the user is already verified, and the
        response wire shape is identical, so a double-clicked
        verification link doesn't show the user an error on the
        second redeem.
        """
        user = user_store.consume_email_verification_token(body.token)
        if user is None:
            log.warning("auth.email_verification.invalid_token")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Verification link is invalid or has expired. "
                    "Request a new one."
                ),
            )
        log.info(
            "auth.email_verification.confirmed",
            user_id=user.id,
            email=user.email,
        )
        return {"ok": True}

    @router.post("/password/change", status_code=200)
    async def change_password(
        body: PasswordChangeBody, request: Request, response: Response
    ) -> dict[str, Any]:
        """Rotate the current user's password.

        Auth contract:

        * Requires a valid session cookie (401 anonymous).
        * Requires re-entry of the current password (403 mismatch).
          Without this, a hijacked session cookie could silently
          rotate the credential — locking the legitimate user out.

        Side effects (in order):

        1. ``update_password`` — bcrypt at the same cost factor as
           signup.
        2. ``revoke_all_sessions_for_user`` — wipes EVERY session,
           including the caller's. Every other device gets logged
           out (defense in depth against compromised cookies on
           other devices).
        3. ``create_session`` — mint a fresh session for the
           current device so the caller stays logged in. The new
           token + new cookie ship back on the response.

        The session rotation matters: without it, an XSS exploit
        that previously captured the cookie would still have a
        valid token after the password change.
        """
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(
                status_code=401, detail="Authentication required."
            )
        if not user_store.verify_password(user.id, body.current_password):
            log.warning(
                "auth.password_change.wrong_current",
                user_id=user.id,
            )
            raise HTTPException(
                status_code=403,
                detail="Current password is incorrect.",
            )
        if not user_store.update_password(user.id, body.new_password):
            # Defensive — the user row's existence is what made the
            # auth gate let us in; update missing it is a deeper bug.
            log.error("auth.password_change.update_failed", user_id=user.id)
            raise HTTPException(
                status_code=500,
                detail="Could not update password. Please try again.",
            )
        revoked = user_store.revoke_all_sessions_for_user(user.id)
        new_session = user_store.create_session(user.id, ttl)
        _set_session_cookie(
            response,
            serializer.dumps(new_session.token),
            cookie_name=cookie_name,
            ttl_seconds=ttl_seconds,
            secure=secure_cookies,
        )
        # CSRF token is rotated alongside the session so an XSS
        # exploit that captured the old one can't fire the next
        # mutation. The cookie's max-age matches the new session.
        _set_csrf_cookie_on(
            response, secure=secure_cookies, ttl_seconds=ttl_seconds
        )
        log.info(
            "auth.password_change.ok",
            user_id=user.id,
            other_sessions_revoked=max(revoked - 1, 0),
        )
        return {"ok": True}

    @router.delete("/account", status_code=200)
    async def delete_account(
        body: AccountDeleteBody, request: Request, response: Response
    ) -> dict[str, Any]:
        """Hard-delete the current account.

        Auth contract:

        * Requires a valid session cookie (401 if anonymous).
        * Requires re-entry of the password in the request body
          (403 on mismatch). Even with a stolen session cookie an
          attacker cannot silently delete the account.

        Side effects (in order):

        1. ``on_account_deleted`` hook fires before the user row
           is removed so the caller (app.py) can purge cross-store
           data without a FK (watchlists live in WatchlistStore
           with no FK back to ``users(id)``).
        2. ``user_store.delete_user`` — the user row goes; FK
           cascades wipe sessions + password-reset tokens + email-
           verification tokens in the same statement.
        3. Session + CSRF cookies cleared on the response so the
           caller's browser doesn't keep stale credentials.

        Idempotent at the "user is already gone" level — if the
        cascade ran but the cookies remained, a second call hits
        401 from the auth gate (no current user) and the client
        moves on.
        """
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if not user_store.verify_password(user.id, body.password):
            log.warning(
                "auth.account_delete.wrong_password",
                user_id=user.id,
            )
            raise HTTPException(
                status_code=403,
                detail="Password is incorrect.",
            )
        # Cross-store purge hook fires first so a hook failure
        # leaves a recoverable state (user row still there, can
        # retry deletion).
        if on_account_deleted is not None:
            try:
                on_account_deleted(user.id)
            except Exception as exc:
                log.error(
                    "auth.account_delete.cross_store_failed",
                    user_id=user.id,
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Could not finish deleting the account. "
                        "Please try again in a moment."
                    ),
                ) from None
        deleted = user_store.delete_user(user.id)
        log.info("auth.account_delete.ok", user_id=user.id, deleted=deleted)
        _clear_session_cookie(
            response, cookie_name=cookie_name, secure=secure_cookies
        )
        _clear_csrf_cookie_on(response, secure=secure_cookies)
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


def _generate_csrf_token() -> str:
    """Mint a fresh CSRF token.

    URL-safe base64, 32 bytes of entropy — same strength as the
    session token. Stateless from the server's POV: the token's role
    is "browser-restricted secret a cross-origin attacker can't
    read," not "value the server can later look up."
    """
    return secrets.token_urlsafe(32)


def _set_csrf_cookie_on(
    response: Response, *, secure: bool, ttl_seconds: int
) -> None:
    """Mint + write a fresh CSRF cookie on ``response``.

    Crucially NOT HttpOnly — the frontend must read it via
    ``document.cookie`` and echo as ``X-CSRF-Token``. SameSite=lax +
    Secure (in prod) bound exposure; the explicit header check is
    the real defense.
    """
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=_generate_csrf_token(),
        max_age=ttl_seconds,
        httponly=False,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _clear_csrf_cookie_on(response: Response, *, secure: bool) -> None:
    """Drop the CSRF cookie. Paired with session clear on logout."""
    response.delete_cookie(
        key=CSRF_COOKIE_NAME,
        path="/",
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
