"""FastAPI app factory exposing the controller's snapshot pipeline.

The factory accepts an injected ``BaseController`` instance so the same
endpoint shape can serve mock data (in tests / local demos) or live
Alpaca data (in production). No business logic lives here — every
field on the response comes from a freshly-fetched
:class:`DashboardSnapshot`.

Endpoints
---------
- ``GET /api/health`` — liveness probe. Cheap; never touches the
  controller. Returns ``{"status": "ok", "tick": <int>}`` where
  ``tick`` is the controller's tick counter at the time of the call.
- ``GET /api/snapshot`` — one-shot fetch. Asks the controller for the
  next snapshot and returns it as JSON via FastAPI's standard encoder
  (dataclasses → dicts, enums → string values, datetimes → ISO 8601
  with timezone). The returned shape mirrors :class:`DashboardSnapshot`
  one-to-one — that dataclass is the source of truth for the contract.

Phase-0 scope notes
-------------------
- Single user, localhost bind. No auth. ``uvicorn`` is intentionally
  not imported here — the runner is a future ``esther serve`` CLI
  command, this module just builds the app object.
- No streaming endpoint yet. The trader-facing dashboard remains the
  Textual TUI; this is a JSON face for the same data.
- No Pydantic response models. The contract is informally documented
  by the dataclasses in :mod:`src.dashboard.state` and the JSON
  encoder's behavior. When the contract stabilizes we can pin
  explicit response models for OpenAPI typing — until then the
  flexibility is worth more than the spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController


def create_app(controller: BaseController) -> FastAPI:
    """Build a FastAPI app bound to a controller.

    The controller is closed over by the route handlers — every
    request fetches a fresh snapshot from this same instance. Pass a
    :class:`MockDashboardController` for tests / demos, a real
    :class:`DashboardController` for live data.
    """
    app = FastAPI(
        title="Esther API",
        description=(
            "Read-only JSON mirror of the Esther dashboard pipeline. "
            "Decision-support data only — this API exposes no order-submission "
            "primitives, and the upstream data layer is paper-feed locked."
        ),
        version="0.1.0",
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
        """
        snap = await controller.fetch_snapshot()
        result: dict[str, Any] = jsonable_encoder(snap)
        return result

    return app
