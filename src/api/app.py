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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from src.api.analyzer import register_analyzer_routes
from src.api.auth import register_auth_routes
from src.api.broker import SnapshotBroker
from src.api.compare import register_compare_routes
from src.api.fundamentals import register_fundamentals_routes
from src.api.health import register_health_routes
from src.api.history import register_history_routes
from src.api.movement import register_movement_routes
from src.api.research import register_research_routes
from src.api.sector import register_sector_routes
from src.api.spotlight import register_spotlight_routes
from src.api.watchlist import register_watchlist_routes
from src.config import get_settings
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
from src.intelligence.calibration import CalibrationStore
from src.intelligence.calibration_worker import (
    CalibrationMaturationWorker,
    bars_cache_price_lookup,
)

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController


# How long the SSE handler waits for a fresh snapshot before emitting
# a keep-alive comment line. Must be smaller than the typical reverse-
# proxy idle timeout (nginx default is 60s, Cloudflare 100s) so a
# slow tick interval doesn't cause the connection to be culled.
_SSE_HEARTBEAT_SECONDS = 15.0


def create_app(
    controller: BaseController,
    *,
    stream_interval: float | None = None,
    cors_origins: list[str] | None = None,
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
        """
        return {"status": "ok", "watchlist_size": len(controller.watchlist)}

    @app.get("/api/snapshot")
    async def snapshot() -> dict[str, Any]:
        """Fetch the next dashboard snapshot and return it as JSON.

        Encoding rules (inherited from FastAPI's ``jsonable_encoder``):

        - ``datetime`` → ISO-8601 with timezone (``2026-05-14T19:42:11+00:00``)
        - ``Enum`` → ``.value`` (so ``SignalAction.BUY`` → ``"buy"``)
        - ``dataclass`` → dict of its fields
        - ``tuple`` → list

        The response shape mirrors :class:`DashboardSnapshot` —
        ``rows`` is the per-symbol watchlist data, top-level fields
        carry the pulse / regime / alerts / events / session-store
        status, and the timestamp marks when the snapshot was built
        (not when the request arrived).

        This endpoint is independent of ``/api/stream`` — it always
        calls ``controller.fetch_snapshot()`` directly rather than
        reading from the broker cache. Two parallel mechanisms keep
        the REST contract simple (request-response) while letting the
        stream endpoint share the broker tick loop.
        """
        snap = await controller.fetch_snapshot()
        result: dict[str, Any] = jsonable_encoder(snap)
        return result

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        """Server-Sent Events stream of dashboard snapshots.

        Wire format::

            : connected\\n\\n                # initial flush + heartbeat
            id: <tick>\\ndata: <json>\\n\\n   # per snapshot
            : keepalive\\n\\n                # every ≤15s when idle

        Browsers / EventSource clients reconnect automatically when
        the connection drops; on reconnect the broker's
        ``subscribe`` immediately re-seeds the queue with the cached
        latest snapshot so the client sees current state without
        waiting up to one full interval.
        """
        broker: SnapshotBroker | None = getattr(app.state, "broker", None)
        if broker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "streaming is disabled on this server (no broker configured). "
                    "Pass stream_interval to create_app() to enable."
                ),
            )

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
                    payload = json.dumps(jsonable_encoder(snap))
                    yield f"id: {snap.tick}\ndata: {payload}\n\n".encode()
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
            if effective_watchlist_store is None or not seed_symbols:
                return
            effective_watchlist_store.seed_default(user.id, seed_symbols)
            # Also nudge the controller so first-tick after signup
            # already has rows for the seeded symbols. add_symbol
            # is idempotent so the typical case (seeded symbols
            # already in the default watchlist) is a no-op.
            for sym in seed_symbols:
                controller.add_symbol(sym)

        register_auth_routes(
            app,
            user_store=effective_user_store,
            session_secret=settings.session_secret_key.get_secret_value(),
            session_ttl_days=settings.session_ttl_days,
            cookie_name=settings.session_cookie_name,
            # Secure cookies require HTTPS. Off in dev (localhost),
            # production deployments should flip ``ESTHER_SECURE_COOKIES=1``
            # — exposed via Settings.app_env once we wire that.
            secure_cookies=False,
            post_signup_hook=(
                _seed_user_watchlist
                if effective_watchlist_store is not None
                else None
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
                )
            try:
                await svc.fetch(symbol)
            except ProviderChainExhausted as exc:
                observed = {
                    h.status for h in exc.health if h.status not in ("skipped", "ok")
                }
                transient = {"rate_limited", "transient"}
                if observed and observed.issubset(transient):
                    raise TransientRetryFailure(tuple(exc.errors)) from None
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=effective_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
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
