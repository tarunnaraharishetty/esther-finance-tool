"""FastAPI app factory exposing the controller's snapshot pipeline.

The factory accepts an injected ``BaseController`` instance so the same
endpoint shape can serve mock data (in tests / local demos) or live
Alpaca data (in production). No business logic lives here — every
field on the response comes from a freshly-fetched
:class:`DashboardSnapshot`.

Endpoints
---------
- ``GET /api/health`` — liveness probe. Cheap; never touches the
  controller pipeline.
- ``GET /api/snapshot`` — one-shot fetch. Asks the controller for a
  fresh snapshot and returns it as JSON via FastAPI's standard encoder
  (dataclasses → dicts, enums → string values, datetimes → ISO 8601
  with timezone). The returned shape mirrors :class:`DashboardSnapshot`
  one-to-one — that dataclass is the source of truth for the contract.
- ``GET /api/stream`` — Server-Sent Events. Emits one event per
  controller tick (cadence: ``stream_interval`` seconds, default
  matches ``Settings.dashboard_refresh_seconds``). Heartbeat comments
  every ≤15s keep proxies from closing idle connections. Reconnect-
  safe: EventSource auto-reconnects, and new subscribers immediately
  receive the cached latest snapshot — no waiting up to one full
  interval for the first paint. Only enabled when ``stream_interval``
  is set on :func:`create_app`; otherwise returns 503.

Phase-0 scope notes
-------------------
- Single user, localhost bind. No auth. ``uvicorn`` is intentionally
  not imported here — the ``esther serve`` CLI is the runner, this
  module just builds the app object.
- No Pydantic response models. The contract is informally documented
  by the dataclasses in :mod:`src.dashboard.state` and the JSON
  encoder's behavior. When the contract stabilizes we can pin
  explicit response models for OpenAPI typing — until then the
  flexibility is worth more than the spec.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from src.api.analyzer import register_analyzer_routes
from src.api.auth import get_current_user, register_auth_routes
from src.api.broker import SnapshotBroker
from src.api.compare import register_compare_routes
from src.api.fundamentals import register_fundamentals_routes
from src.api.health import register_health_routes
from src.api.history import register_history_routes
from src.api.middleware import (
    InMemoryRateLimiter,
    build_rate_limiter_dep,
    install_auth_gate,
    install_csrf_gate,
    install_path_rate_limits,
    install_request_log,
)
from src.api.movement import register_movement_routes
from src.api.research import register_research_routes
from src.api.sector import register_sector_routes
from src.api.spotlight import register_spotlight_routes
from src.api.watchlist import register_watchlist_routes
from src.config import get_settings
from src.dashboard.state import filter_snapshot_for_symbols
from src.data.accuracy_store import AccuracyStore
from src.data.health_store import HealthStore
from src.data.retry_queue import BackoffPolicy, RetryQueue
from src.data.retry_worker import (
    PermanentRetryFailure,
    RetryWorker,
    TransientRetryFailure,
)
from src.data.user_store import User, UserStore
from src.data.watchlist_store import WatchlistStore
from src.intelligence.analyzer.sector_medians import SectorCohort
from src.intelligence.calibration import CalibrationStore
from src.intelligence.calibration_worker import (
    CalibrationMaturationWorker,
    bars_cache_price_lookup,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController


# How long the SSE handler waits for a fresh snapshot before emitting
# a keep-alive comment line. Must be smaller than the typical reverse-
# proxy idle timeout (nginx default is 60s, Cloudflare 100s) so a
# slow tick interval doesn't cause the connection to be culled.
_SSE_HEARTBEAT_SECONDS = 15.0


# Module-level dependency alias. FastAPI / pydantic resolve OpenAPI
# schemas eagerly at first request; aliases declared inside the
# ``create_app`` function body remain ForwardRefs and trip
# ``PydanticUserError: TypeAdapter[...] is not fully defined``. Lifting
# the alias to module scope keeps the dependency typing readable on
# each route signature without poisoning the OpenAPI generator.
OptionalUser = Annotated[User | None, Depends(get_current_user)]


def create_app(
    controller: BaseController,
    *,
    stream_interval: float | None = None,
    cors_origins: list[str] | None = None,
    trusted_hosts: list[str] | None = None,
    frontend_dir: Path | None = None,
    fundamentals_service: Any = None,
    fundamentals_cache_dir: Path | None = None,
    analyzer_cache_dir: Path | None = None,
    health_store: HealthStore | None = None,
    retry_queue: RetryQueue | None = None,
    retry_worker_enabled: bool | None = None,
    calibration_store: CalibrationStore | None = None,
    calibration_maturation_enabled: bool | None = None,
    research_cache_dir: Path | None = None,
    compare_narrative_cache_dir: Path | None = None,
    sector_cache_dir: Path | None = None,
    movement_cache_dir: Path | None = None,
    spotlight_cache_dir: Path | None = None,
    accuracy_store: AccuracyStore | None = None,
    user_store: UserStore | None = None,
    watchlist_store: WatchlistStore | None = None,
    default_watchlist_symbols: list[str] | None = None,
    sector_cohort: SectorCohort | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to a controller.

    The controller is closed over by the route handlers — every
    request fetches a fresh snapshot from this same instance.

    Args:
        controller: The shared :class:`BaseController`. Used by every
            endpoint; same instance for REST and streaming.
        stream_interval: If set, a :class:`SnapshotBroker` is started
            during the FastAPI lifespan and the ``/api/stream``
            endpoint is enabled. If ``None`` (the default), no broker
            is started — useful in tests that only exercise the REST
            endpoints. The ``esther serve`` CLI passes
            ``Settings.dashboard_refresh_seconds`` here.
        cors_origins: Explicit allow-list of origins for cross-origin
            requests. ``None`` falls back to the safe default
            ``["http://localhost:5173", "http://127.0.0.1:5173"]``
            (the Vite dev server). In production same-origin deploys
            CORS is never invoked by the browser, so the value is
            irrelevant — but we always wire the middleware for parity.
            Pass ``[]`` to disable cross-origin entirely.
        frontend_dir: If set and the directory exists, mount the Vite
            build at the app's root: ``/assets/*`` serves hashed bundle
            files and any unmatched non-``/api`` route returns
            ``index.html``. The ``esther serve`` CLI passes
            ``settings.project_root / "web" / "dist"``. ``None`` or a
            non-existent path = API-only mode (the dev workflow, where
            Vite serves the frontend on its own port).
    """

    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        """Start workers on startup; stop on shutdown.

        Three workers, all opt-in:

        - The snapshot broker spins up when ``stream_interval`` is
          positive (drives the SSE endpoint).
        - The retry worker spins up when a retry queue exists AND
          ``retry_queue_worker_enabled`` is True. Drains transient
          fundamentals failures on its own cadence.
        - The calibration maturation worker spins up when the
          calibration store exists AND
          ``calibration_maturation_enabled`` is True. Settles matured
          observations against the bars cache.

        Every worker is attached to ``app.state`` so route closures
        can find them; every worker is cancelled on shutdown.
        """
        broker: SnapshotBroker | None = None
        if stream_interval is not None:
            broker = SnapshotBroker(controller, interval=stream_interval)
            await broker.start()
        app_.state.broker = broker

        # ``starlette.datastructures.State`` stores attributes in
        # ``_state``, not ``__dict__`` — the lifespan reads via
        # ``getattr`` so attributes set on ``app.state`` from outside
        # the lifespan (where the workers are constructed) round-trip
        # correctly. Direct ``__dict__.get`` returns None and silently
        # disables the worker.
        worker: RetryWorker | None = getattr(app_.state, "retry_worker", None)
        if worker is not None:
            await worker.start()

        cal_worker: CalibrationMaturationWorker | None = getattr(
            app_.state, "calibration_worker", None
        )
        if cal_worker is not None:
            await cal_worker.start()

        try:
            yield
        finally:
            if broker is not None:
                await broker.stop()
            app_.state.broker = None
            if worker is not None:
                await worker.stop()
                app_.state.retry_worker = None
            if cal_worker is not None:
                await cal_worker.stop()
                app_.state.calibration_worker = None

    app = FastAPI(
        title="Esther API",
        description=(
            "Read-only JSON mirror of the Esther dashboard pipeline. "
            "Decision-support data only — this API exposes no order-submission "
            "primitives, and the upstream data layer is paper-feed locked."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Liveness probe.

        Does not call ``fetch_snapshot`` so a stalled pipeline can't
        hang the health endpoint. The watchlist length is the only
        controller field we read here — cheap and useful for
        confirming the service is bound to the expected instance.

        K8s-style separation: this endpoint says "the process is up
        and the FastAPI app is wired"; readiness (DB connectivity,
        downstream reachability) lives on ``/api/readyz`` below.
        """
        return {"status": "ok", "watchlist_size": len(controller.watchlist)}

    @app.get("/api/livez")
    async def livez() -> dict[str, Any]:
        """K8s-convention alias for ``/api/health``.

        Kept as a separate route (rather than 301-redirecting) so a
        liveness probe never relies on the client following a
        redirect — some prober configurations treat 3xx as failure.
        Same payload; both routes stay public for cluster operators
        who follow the livez/readyz convention.
        """
        return {"status": "ok", "watchlist_size": len(controller.watchlist)}

    @app.get("/api/readyz")
    async def readyz() -> dict[str, Any]:
        """Readiness probe — distinguish "wired" from "ready to serve."

        Verifies each persistent dependency the process can't function
        without. K8s / load balancers should remove an instance from
        the rotation when this fails; the process stays alive (so
        ``/api/health`` keeps passing) but no new requests are routed
        to it until the underlying issue clears.

        Checks are best-effort and bounded: each backend has a cheap
        read it can answer in milliseconds. A 503 response carries a
        ``checks`` map so operators can see which dependency degraded.
        """
        checks: dict[str, str] = {}
        ok = True

        users: UserStore | None = getattr(app.state, "user_store", None)
        if users is not None:
            try:
                # Indexed SELECT on the unique email column — cheap,
                # verifies the file is reachable AND the schema is in
                # place. The probe email is impossible (no ``@``) so
                # the row never matches; we only care that the query
                # ran without an OperationalError.
                users.get_user_by_email("__readyz_probe__")
                checks["user_store"] = "ok"
            except Exception as exc:
                ok = False
                checks["user_store"] = f"error: {type(exc).__name__}"
        else:
            checks["user_store"] = "disabled"

        rq: RetryQueue | None = getattr(app.state, "retry_queue", None)
        if rq is not None:
            try:
                # ``due_entries`` is the worker's hot path — exercising
                # it as the probe means a regression that breaks it
                # also degrades readiness. The (status, next_attempt_at)
                # composite index keeps this cheap even at scale.
                rq.due_entries()
                checks["retry_queue"] = "ok"
            except Exception as exc:
                ok = False
                checks["retry_queue"] = f"error: {type(exc).__name__}"
        else:
            checks["retry_queue"] = "disabled"

        # Watchlist store shares the file with UserStore by default
        # but has its own table + migrations trail — probe it
        # independently so a schema break on the watchlist side
        # doesn't hide behind a healthy users probe.
        wl: WatchlistStore | None = getattr(app.state, "watchlist_store", None)
        if wl is not None:
            try:
                # ``list_for(0)`` returns ``[]`` because no real user
                # has id 0 — but it exercises the table and the
                # ordering index. Cheap regression probe.
                wl.list_for(0)
                checks["watchlist_store"] = "ok"
            except Exception as exc:
                ok = False
                checks["watchlist_store"] = f"error: {type(exc).__name__}"
        else:
            checks["watchlist_store"] = "disabled"

        body: dict[str, Any] = {
            "status": "ok" if ok else "degraded",
            "checks": checks,
        }
        if not ok:
            return JSONResponse(body, status_code=503)  # type: ignore[return-value]
        return body

    def _user_allowed_symbols(user: User | None) -> frozenset[str] | None:
        """Resolve the user's watchlist symbols for per-user snapshot filtering.

        Three cases:

        * ``user`` is ``None`` → no auth wired (single-user dev / tests).
          Return ``None`` so the snapshot passes through unfiltered.
        * ``user`` is set but the WatchlistStore is not wired → return
          an empty set so the user sees no per-symbol rows (safer than
          leaking the full universe).
        * ``user`` is set and a WatchlistStore is available → return
          the user's symbols as an upper-cased ``frozenset``.
        """
        if user is None:
            return None
        wl_store: WatchlistStore | None = getattr(app.state, "watchlist_store", None)
        if wl_store is None:
            return frozenset()
        return frozenset(e.symbol.upper() for e in wl_store.list_for(user.id))

    @app.get("/api/snapshot")
    async def snapshot(user: OptionalUser) -> dict[str, Any]:
        """Fetch the next dashboard snapshot, filtered to the caller's watchlist.

        Auth contract:

        * When a UserStore is wired (the production path), this route
          requires a valid session cookie — anonymous requests get 401.
        * When auth is disabled at the app level (dev workflow without
          a UserStore), the snapshot flows through unfiltered.

        Per-user filtering removes rows / alerts / opportunities for
        symbols outside the caller's watchlist; aggregate fields
        (pulse, regime, session-store health, events) pass through
        because they describe the operating environment, not any
        individual user's positions.
        """
        if getattr(app.state, "user_store", None) is not None and user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        snap = await controller.fetch_snapshot()
        filtered = filter_snapshot_for_symbols(snap, _user_allowed_symbols(user))
        result: dict[str, Any] = jsonable_encoder(filtered)
        return result

    @app.get("/api/stream")
    async def stream(
        request: Request, user: OptionalUser
    ) -> StreamingResponse:
        """Server-Sent Events stream of dashboard snapshots.

        Same auth contract as ``/api/snapshot`` — when a UserStore is
        wired the route requires a valid session cookie; without one
        the response is 401. Each emitted snapshot is filtered down
        to the user's watchlist symbols (see
        :func:`filter_snapshot_for_symbols`) so two users on the same
        server can't read each other's rows.

        Wire format::

            : connected\\n\\n                # initial flush + heartbeat
            id: <tick>\\ndata: <json>\\n\\n   # per snapshot
            : keepalive\\n\\n                # every ≤15s when idle
        """
        if getattr(app.state, "user_store", None) is not None and user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")

        broker: SnapshotBroker | None = getattr(app.state, "broker", None)
        if broker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "streaming is disabled on this server (no broker configured). "
                    "Pass stream_interval to create_app() to enable."
                ),
            )

        # Resolve the user's allowed symbols ONCE per subscription so
        # the SSE generator doesn't re-query the watchlist store on
        # every tick. A future enhancement would invalidate this on
        # /api/watchlist mutations from the same session; for v1, a
        # reconnect (which the EventSource does on watchlist add/remove)
        # re-resolves naturally.
        allowed = _user_allowed_symbols(user)

        async def event_generator() -> AsyncIterator[bytes]:
            queue = broker.subscribe()
            try:
                # Flush headers immediately so the client knows the
                # connection is live before the first tick lands.
                # Comment-line events are ignored by EventSource but
                # still cause the response to start streaming through
                # any reverse proxy in front of us.
                yield b": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        snap = await asyncio.wait_for(
                            queue.get(), timeout=_SSE_HEARTBEAT_SECONDS
                        )
                    except TimeoutError:
                        # No tick within the keep-alive window —
                        # send a comment to keep the connection warm.
                        yield b": keepalive\n\n"
                        continue
                    filtered = filter_snapshot_for_symbols(snap, allowed)
                    payload = json.dumps(jsonable_encoder(filtered))
                    yield f"id: {filtered.tick}\ndata: {payload}\n\n".encode()
            finally:
                broker.unsubscribe(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                # Belt-and-braces against intermediary caching. Some
                # corporate proxies buffer responses unless this is
                # explicit.
                "Cache-Control": "no-cache",
                # nginx-specific hint: disable response buffering on
                # this endpoint so events flush in real time.
                "X-Accel-Buffering": "no",
            },
        )

    # Research-thesis routes (cached, structured per-symbol research
    # reports). Cache dir lives under the project's data/ tree so it
    # rotates with the rest of the runtime state and stays out of the
    # source tree.
    settings = get_settings()
    research_cache = (
        research_cache_dir
        if research_cache_dir is not None
        else settings.project_root / "data" / "research_cache"
    )
    register_research_routes(app, controller, cache_dir=research_cache)

    # Health observability — only wire the route when a HealthStore is
    # available. When `health_store` is None we look up the settings
    # default; passing `health_store_path=None` in settings disables
    # the store entirely (and the route 503s on access).
    effective_health_store: HealthStore | None = health_store
    if effective_health_store is None and settings.health_store_path is not None:
        effective_health_store = HealthStore(settings.health_store_path)

    # Accuracy ledger: shares the same SQLite file as the health store
    # so operators have a single backup target. Lazy schema init means
    # the table only materializes the first time reconciliation fires
    # — tests that never exercise it pay no I/O cost. Tests can also
    # inject their own instance via ``accuracy_store=`` to align the
    # test's accuracy ledger with the app's view.
    effective_accuracy_store: AccuracyStore | None = accuracy_store
    if (
        effective_accuracy_store is None
        and settings.health_store_path is not None
    ):
        effective_accuracy_store = AccuracyStore(settings.health_store_path)

    # User store + auth routes. Same opt-out shape as the others —
    # ``USER_STORE_PATH=`` in env disables auth entirely (the
    # dev/single-user workflow that pre-dates accounts). When a
    # store IS wired, /api/auth/* mounts and the session-aware
    # dependencies become available on every request.
    effective_user_store: UserStore | None = user_store
    if (
        effective_user_store is None
        and settings.user_store_path is not None
    ):
        effective_user_store = UserStore(settings.user_store_path)

    # WatchlistStore shares the users.db file when one is configured
    # (per-user data colocated with accounts). Independent connection;
    # WAL handles cross-store concurrency. ``None`` when auth itself
    # is disabled — no users, no watchlists.
    effective_watchlist_store: WatchlistStore | None = watchlist_store
    if (
        effective_watchlist_store is None
        and effective_user_store is not None
    ):
        effective_watchlist_store = WatchlistStore(
            effective_user_store.db_path
        )

    if effective_user_store is not None:
        # Closures for the auth post-signup hook and the watchlist
        # add hook. Both reach into module-scope objects (the
        # watchlist store, the controller), but staying anonymous
        # here keeps the wiring local to app.py.
        seed_symbols = tuple(
            default_watchlist_symbols
            if default_watchlist_symbols is not None
            else settings.default_watchlist_symbols
        )

        def _seed_user_watchlist(user: User) -> None:
            """Seed the new user's watchlist with the default symbols.

            Transactional contract (B-14): if the seed fails the user
            row is deleted before re-raising so signup as a whole
            either succeeds with a populated watchlist or leaves no
            trace. The previous best-effort behavior produced an
            account whose first dashboard view was empty — users
            reasonably concluded auth was broken.

            Watchlist rows have no FK to ``users(id)`` (see
            ``watchlist_store.py``), so the rollback explicitly purges
            any rows the seed managed to write before failing.
            """
            if effective_watchlist_store is None or not seed_symbols:
                return
            try:
                effective_watchlist_store.seed_default(user.id, seed_symbols)
                # Nudge the controller so the first tick after signup
                # already has rows for the seeded symbols. ``add_symbol``
                # is idempotent so the typical case is a no-op.
                for sym in seed_symbols:
                    controller.add_symbol(sym)
            except Exception:
                # Rollback in the reverse order of acquisition: drop
                # any partially-seeded watchlist rows, then the user.
                # Each cleanup step is guarded so a secondary failure
                # in the rollback path still lets the original error
                # surface to the caller (and the logs).
                for sym in seed_symbols:
                    try:
                        effective_watchlist_store.remove(user.id, sym)
                    except Exception as inner:
                        log.warning(
                            "auth.signup.rollback.watchlist_remove_failed",
                            user_id=user.id,
                            symbol=sym,
                            error=str(inner),
                        )
                try:
                    effective_user_store.delete_user(user.id)
                except Exception as inner:
                    log.warning(
                        "auth.signup.rollback.user_delete_failed",
                        user_id=user.id,
                        error=str(inner),
                    )
                raise

        # Per-IP rate limiters for the auth write routes. Four
        # separate buckets, not one shared "auth" budget — the abuse
        # patterns differ enough that one shared cap can't serve
        # them all well:
        #
        # * Login: brute-force / credential-stuffing tries dozens of
        #   passwords against one or many accounts. The right shape
        #   is a moderate cap in a short window so a forgetful human
        #   isn't punished. 10 login / IP / 5 min.
        #
        # * Signup: account-creation flooding (typically for free-tier
        #   abuse). Real humans sign up ~once. The right shape is a
        #   strict cap in a long window. 3 signup / IP / hour.
        #
        # * Password-reset request: similar shape to signup — real
        #   users hit it rarely, scripted abuse hits hard. Slightly
        #   higher headroom than signup (a forgetful user might miss
        #   the email and retry) but still tight. 5 / IP / hour.
        #
        # * Password-reset confirm: protects against brute-forcing the
        #   token in the URL. The 256-bit entropy makes this
        #   infeasible anyway, but a cap forces a noticeably slow
        #   attempt rate. 20 / IP / 5 min.
        login_rate_limiter = InMemoryRateLimiter(
            max_calls=10, per_seconds=300.0
        )
        signup_rate_limiter = InMemoryRateLimiter(
            max_calls=3, per_seconds=3600.0
        )
        pwreset_request_rate_limiter = InMemoryRateLimiter(
            max_calls=5, per_seconds=3600.0
        )
        pwreset_confirm_rate_limiter = InMemoryRateLimiter(
            max_calls=20, per_seconds=300.0
        )
        app.state.login_rate_limiter = login_rate_limiter
        app.state.signup_rate_limiter = signup_rate_limiter
        app.state.pwreset_request_rate_limiter = pwreset_request_rate_limiter
        app.state.pwreset_confirm_rate_limiter = pwreset_confirm_rate_limiter

        # Emailer for the password-reset request route. When
        # ``SMTP_HOST`` is configured we wire the real transport;
        # otherwise the auth route falls back to its own
        # LogPasswordResetEmailer default and reset links land in the
        # structured log.
        from src.api.password_reset_emailer import (
            PasswordResetEmailer,
            SmtpPasswordResetEmailer,
        )

        pw_emailer: PasswordResetEmailer | None = None
        if settings.smtp_host:
            pw_emailer = SmtpPasswordResetEmailer(
                host=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_username,
                password=(
                    settings.smtp_password.get_secret_value()
                    if settings.smtp_password is not None
                    else ""
                ),
                sender=settings.smtp_sender,
                timeout_seconds=settings.smtp_timeout_seconds,
            )

        register_auth_routes(
            app,
            user_store=effective_user_store,
            session_secret=settings.session_secret_key.get_secret_value(),
            session_ttl_days=settings.session_ttl_days,
            cookie_name=settings.session_cookie_name,
            # Source: ``Settings.secure_cookies``. Defaults to True;
            # the model validator refuses False in non-dev. Override
            # via ``SECURE_COOKIES=false`` only for local HTTP testing.
            secure_cookies=settings.secure_cookies,
            post_signup_hook=(
                _seed_user_watchlist
                if effective_watchlist_store is not None
                else None
            ),
            login_rate_limit_dep=build_rate_limiter_dep(
                login_rate_limiter, scope="auth_login"
            ),
            signup_rate_limit_dep=build_rate_limiter_dep(
                signup_rate_limiter, scope="auth_signup"
            ),
            password_reset_emailer=pw_emailer,
            password_reset_ttl=timedelta(hours=settings.password_reset_ttl_hours),
            password_reset_base_url=settings.password_reset_base_url,
            password_reset_request_rate_limit_dep=build_rate_limiter_dep(
                pwreset_request_rate_limiter, scope="auth_pwreset_request"
            ),
            password_reset_confirm_rate_limit_dep=build_rate_limiter_dep(
                pwreset_confirm_rate_limiter, scope="auth_pwreset_confirm"
            ),
        )

        if effective_watchlist_store is not None:

            def _track_in_controller(sym: str) -> None:
                # Closure over the bound controller. Idempotent —
                # add_symbol returns False on duplicate.
                controller.add_symbol(sym)

            register_watchlist_routes(
                app,
                watchlist_store=effective_watchlist_store,
                on_symbol_added=_track_in_controller,
            )

    # Retry queue: same opt-out shape. ``retry_queue_path=None`` in
    # settings disables persistence; otherwise we wire one up.
    effective_retry_queue: RetryQueue | None = retry_queue
    if effective_retry_queue is None and settings.retry_queue_path is not None:
        effective_retry_queue = RetryQueue(
            settings.retry_queue_path,
            backoff=BackoffPolicy(
                initial_seconds=settings.retry_queue_initial_backoff_seconds,
                max_seconds=settings.retry_queue_max_backoff_seconds,
            ),
            max_attempts=settings.retry_queue_max_attempts,
        )
    # Stash on app.state so /api/readyz can probe it.
    app.state.retry_queue = effective_retry_queue

    # Calibration store: same opt-out shape. Lazy init means
    # constructing the store doesn't touch disk until the analyzer
    # actually queries it.
    effective_calibration_store: CalibrationStore | None = calibration_store
    if (
        effective_calibration_store is None
        and settings.calibration_store_path is not None
    ):
        effective_calibration_store = CalibrationStore(
            settings.calibration_store_path
        )

    # Provider trust composer — wires both stores so the providers
    # health endpoint can surface trust_breakdown + field_accuracy
    # per provider. Falls back gracefully when either store is None
    # (cold-start defaults to weight = 1.0 across the board).
    from src.intelligence.fundamentals.provider_trust import ProviderTrust as _PT

    effective_provider_trust = _PT(
        accuracy_store=effective_accuracy_store,
        health_store=effective_health_store,
    )

    if effective_health_store is not None:
        register_health_routes(
            app,
            store=effective_health_store,
            retry_queue=effective_retry_queue,
            accuracy_store=effective_accuracy_store,
            provider_trust=effective_provider_trust,
        )

    # Sector-medians cohort: observes every successful fundamentals
    # fetch and recomputes the live medians table that ``lookup()``
    # serves to the valuation ensemble (B-13). One instance per app —
    # tests can inject their own to assert behavior in isolation.
    effective_sector_cohort: SectorCohort = (
        sector_cohort if sector_cohort is not None else SectorCohort()
    )
    app.state.sector_cohort = effective_sector_cohort

    # Fundamentals routes (Phase 2). Independent of the controller —
    # the fundamentals chain only needs settings + the orchestrator.
    # Tests inject ``fundamentals_service`` to avoid network; production
    # passes ``None`` and lets the route module build one lazily.
    fundamentals_cache = (
        fundamentals_cache_dir
        if fundamentals_cache_dir is not None
        else settings.project_root / "data" / "fundamentals_cache"
    )
    register_fundamentals_routes(
        app,
        cache_dir=fundamentals_cache,
        service=fundamentals_service,
        health_store=effective_health_store,
        retry_queue=effective_retry_queue,
        accuracy_store=effective_accuracy_store,
        sector_cohort=effective_sector_cohort,
    )

    # Analyzer routes (Phase 6) — assemble technicals + valuation +
    # grounded explanation into one cached endpoint. ``analyzer_cache_dir``
    # is overridable for tests so per-call state doesn't leak across
    # test cases via the on-disk cache.
    analyzer_cache = (
        analyzer_cache_dir
        if analyzer_cache_dir is not None
        else settings.project_root / "data" / "analyzer_cache"
    )
    register_analyzer_routes(
        app,
        controller,
        cache_dir=analyzer_cache,
        fundamentals_service=fundamentals_service,
        health_store=effective_health_store,
        retry_queue=effective_retry_queue,
        calibration_store=effective_calibration_store,
        accuracy_store=effective_accuracy_store,
    )

    # Compare endpoint reads the analyzer assembler off app.state, so
    # this registration must come *after* the analyzer routes are wired.
    # The narrative cache lives under data/ alongside the other on-disk
    # caches so it rotates with the rest of the runtime state.
    compare_narrative_cache = (
        compare_narrative_cache_dir
        if compare_narrative_cache_dir is not None
        else settings.project_root / "data" / "compare_narrative_cache"
    )
    register_compare_routes(app, narrative_cache_dir=compare_narrative_cache)

    # Sector ranking endpoint — also reads the analyzer assembler off
    # app.state, so it must be registered after the analyzer routes.
    sector_cache = (
        sector_cache_dir
        if sector_cache_dir is not None
        else settings.project_root / "data" / "sector_cache"
    )
    register_sector_routes(app, controller, cache_dir=sector_cache)

    # Key Drivers endpoint — composes analyzer + sector + row, so it
    # must be registered after both the analyzer and sector routes.
    movement_cache = (
        movement_cache_dir
        if movement_cache_dir is not None
        else settings.project_root / "data" / "movement_cache"
    )
    register_movement_routes(app, controller, cache_dir=movement_cache)

    # Watchlist Driver Spotlight — composes per-symbol drivers across
    # the whole watchlist into one fleet-level ranked view. Must be
    # registered after the analyzer + movement routes (it reuses both).
    spotlight_cache = (
        spotlight_cache_dir
        if spotlight_cache_dir is not None
        else settings.project_root / "data" / "spotlight_cache"
    )
    register_spotlight_routes(app, controller, cache_dir=spotlight_cache)

    # Per-symbol historical outcomes drill-down. Only registered when
    # the calibration store exists — when calibration is disabled,
    # the endpoint silently 404s (no data to surface).
    if effective_calibration_store is not None:
        register_history_routes(app, store=effective_calibration_store)

    # Optional retry worker. Wired into ``app.state`` so the lifespan
    # handler picks it up; stays None when disabled. We don't import
    # FundamentalsService at module top to keep this file's import
    # graph minimal — late import here is cheap (one-shot at app
    # construction).
    worker_on = (
        retry_worker_enabled
        if retry_worker_enabled is not None
        else settings.retry_queue_worker_enabled
    )
    if worker_on and effective_retry_queue is not None:
        from src.intelligence.fundamentals import (
            FundamentalsService,
            ProviderChainExhausted,
        )

        async def _retry_fundamentals(symbol: str) -> None:
            """Worker retry hook for a single symbol.

            Translates the fundamentals chain outcome into the
            retry-worker exception protocol: success returns
            normally; transient-only exhaustion raises
            :class:`TransientRetryFailure`; any other failure raises
            :class:`PermanentRetryFailure`.
            """
            svc: FundamentalsService
            if fundamentals_service is not None:
                svc = fundamentals_service  # injected (tests)
            else:
                svc = FundamentalsService.from_settings(
                    settings,
                    health_store=effective_health_store,
                    retry_queue=effective_retry_queue,
                    sector_cohort=effective_sector_cohort,
                )
            try:
                await svc.fetch(symbol)
            except ProviderChainExhausted as exc:
                observed = {
                    h.status for h in exc.health if h.status not in ("skipped", "ok")
                }
                transient = {"rate_limited", "transient"}
                if observed and observed.issubset(transient):
                    # Forward the server-supplied ``Retry-After`` hint so the
                    # queue's reschedule honors RFC 6585 on the second
                    # attempt too — not just on the initial enqueue.
                    raise TransientRetryFailure(
                        tuple(exc.errors),
                        retry_after_seconds=exc.retry_after_seconds,
                    ) from None
                raise PermanentRetryFailure(tuple(exc.errors)) from None

        worker = RetryWorker(
            effective_retry_queue,
            _retry_fundamentals,
            interval_seconds=settings.retry_queue_worker_interval_seconds,
        )
        app.state.retry_worker = worker
    else:
        app.state.retry_worker = None

    # Calibration maturation worker — same lifespan pattern. Settles
    # matured observations against the bars cache (no Alpaca fallback
    # at this stage — production deployments rely on the snapshot loop
    # to keep the cache fresh for active watchlist symbols). Off by
    # default; ``Settings.calibration_maturation_enabled`` flips it on.
    cal_worker_on = (
        calibration_maturation_enabled
        if calibration_maturation_enabled is not None
        else settings.calibration_maturation_enabled
    )
    if cal_worker_on and effective_calibration_store is not None:
        cal_worker = CalibrationMaturationWorker(
            effective_calibration_store,
            bars_cache_price_lookup,
            interval_seconds=settings.calibration_maturation_interval_seconds,
        )
        app.state.calibration_worker = cal_worker
    else:
        app.state.calibration_worker = None

    # CORS for the read-only API. In production the bundled frontend
    # is served from the same origin as the API (see frontend_dir
    # below), so browsers never make CORS requests and this list is
    # irrelevant. The default — Vite's dev server on :5173 — keeps
    # `npm run dev` working out of the box. Override via the
    # ``CORS_ORIGINS`` env var (comma-separated or JSON list).
    effective_origins = cors_origins if cors_origins is not None else [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    # Per-IP throttling for the LLM-backed endpoints. Anthropic costs
    # real money and the prompts are long; bound the worst case at
    # ~30 calls per IP per 5 minutes across the three surfaces. Active
    # users won't notice; scripted abuse will.
    llm_route_limiter = InMemoryRateLimiter(max_calls=30, per_seconds=300.0)
    app.state.llm_route_limiter = llm_route_limiter
    install_path_rate_limits(
        app,
        rules=[
            ("/api/research/", "llm", llm_route_limiter),
            ("/api/analyzer/", "llm", llm_route_limiter),
            ("/api/compare/", "llm", llm_route_limiter),
        ],
    )

    # CSRF gate next. Stays a pass-through unless ``csrf_enabled`` is
    # flipped on app.state — same opt-out shape as auth, so the dev
    # workflow that has no UserStore doesn't suddenly require a token.
    # Enforcement scope: POST/PUT/PATCH/DELETE to /api/* (except
    # /api/auth/* and /api/health) must echo the esther_csrf cookie
    # back in an X-CSRF-Token header.
    app.state.csrf_enabled = effective_user_store is not None
    install_csrf_gate(app)

    # AuthGate next so CORS (added below) wraps it — that way 401
    # responses still carry the right CORS headers and the browser
    # surfaces the rejection cleanly. Pass-through when no UserStore
    # is wired (the dev workflow that pre-dates accounts).
    install_auth_gate(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=effective_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # Stash the proxy-trust flag so ``_client_ip`` can consult it
    # without a circular settings import in the middleware module.
    # Defaults to False — operators behind a reverse proxy must flip
    # ``TRUST_PROXY_HEADERS=true`` explicitly.
    app.state.trust_proxy_headers = settings.trust_proxy_headers

    # Request-log middleware installed after the auth/CSRF/rate-limit
    # gates so it sits OUTSIDE them in the request flow — that way the
    # log line captures the final status the client sees, including
    # rejections from AuthGate (401), CSRF (403), and PathRateLimit
    # (429). It also binds the request id into structlog contextvars
    # so every nested log call inside the request inherits the
    # correlation id automatically.
    install_request_log(app)

    # TrustedHostMiddleware MUST be installed last so it sits at the
    # very outermost edge — a wrong Host should fail with 400 before
    # any other middleware (including the request log) even sees the
    # request, since serving content for an unexpected hostname is
    # exactly the cache-poisoning / password-reset-forgery risk we
    # want to neutralize. The settings model validator already
    # refuses ``["*"]`` in non-dev so a typo doesn't ship.
    effective_trusted_hosts = (
        trusted_hosts if trusted_hosts is not None else settings.trusted_hosts
    )
    if effective_trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=effective_trusted_hosts,
        )

    # Static frontend mount — production same-origin path.
    #
    # When `web/dist/` exists (produced by `npm run build`), FastAPI
    # serves the Vite bundle:
    #   - /assets/*     → hashed JS/CSS chunks
    #   - /favicon.svg  → root-level static files (favicon, robots.txt, ...)
    #   - everything-else-not-/api → index.html (SPA-style fallback,
    #     even though Esther doesn't currently use client-side routing —
    #     leaves the door open without costing anything today)
    # Routes are registered AFTER the API routes above so /api/* wins.
    if frontend_dir is not None and frontend_dir.exists():
        _mount_frontend(app, frontend_dir)

    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Wire the static-file + SPA-fallback routes for ``dist``.

    Split out so ``create_app`` stays focused on the API surface and
    the static-serving logic has a clear seam for future hardening
    (e.g. cache-control headers on the bundle).
    """
    assets_dir = dist / "assets"
    if assets_dir.exists():
        # Mount the hashed-bundle directory. Vite emits filenames like
        # ``index-BXzIQ50v.js`` so these can be cached forever; we let
        # the host (Railway/Render) decide cache headers since they
        # vary by deployment shape.
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    index_html = dist / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        """Serve a root-level static file if one exists; else index.html.

        404s any unmatched ``/api/*`` so the SPA fallback doesn't swallow
        what would otherwise be a clean API miss.
        """
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = dist / full_path
        # Only serve files directly under dist (not directories — those
        # fall through to index.html).
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)


# Re-export the contract for callers that need to type-annotate it
# without importing SnapshotBroker directly.
__all__ = ["create_app"]
