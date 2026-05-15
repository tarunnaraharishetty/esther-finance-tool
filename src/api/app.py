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

from src.api.broker import SnapshotBroker

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
        """Start the snapshot broker on app startup, stop on shutdown.

        Only spins up if a positive ``stream_interval`` was passed.
        The broker is attached to ``app.state.broker`` so the SSE
        handler can find it (route closures can't easily share state
        any other way in FastAPI).
        """
        broker: SnapshotBroker | None = None
        if stream_interval is not None:
            broker = SnapshotBroker(controller, interval=stream_interval)
            await broker.start()
        app_.state.broker = broker
        try:
            yield
        finally:
            if broker is not None:
                await broker.stop()
            app_.state.broker = None

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
