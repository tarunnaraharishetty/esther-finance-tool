"""Per-symbol historical signal outcomes API.

One endpoint:

* ``GET /api/history/{symbol}`` — read the calibration store, filter to
  ``symbol``, bucket by score × outcome × value-range, return Wilson-
  bounded hit rates per bucket. Mirrors the per-symbol view's
  :class:`~src.intelligence.historical_outcomes.HistoricalOutcomes`
  dataclass.

Wired when a ``CalibrationStore`` is available (same opt-in shape as
``/api/health/queue``). When the store is disabled at the deployment
level, the route is never registered and the path 404s.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.config import get_settings
from src.intelligence.calibration import CalibrationStore
from src.intelligence.historical_outcomes import (
    outcomes_to_wire,
    symbol_outcomes,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def register_history_routes(
    app: FastAPI,
    *,
    store: CalibrationStore,
) -> None:
    """Attach the per-symbol historical-outcomes route.

    Args:
        app: FastAPI app to extend.
        store: The same :class:`CalibrationStore` the snapshot
            recorder writes to. Passing a different instance would
            silently disagree about which observations are visible
            here — use the wiring in :func:`src.api.app.create_app`
            so there's one store per app.
    """
    settings = get_settings()

    @app.get("/api/history/{symbol}")
    async def history(
        symbol: str,
        horizon: int | None = None,
        bucket_width: float | None = None,
        min_observations: int | None = None,
    ) -> dict[str, Any]:
        """Return per-symbol observed-outcomes view for ``symbol``.

        Query params:

        * ``horizon`` — outcome horizon in days. Defaults to
          ``Settings.calibration_horizon_days``.
        * ``bucket_width`` — score-bucket width. Defaults to
          ``Settings.calibration_bucket_width``.
        * ``min_observations`` — per-bucket publish threshold.
          Defaults to 5 (lower than pooled — per-symbol N is
          naturally smaller).
        """
        if not symbol or not symbol.strip():
            raise HTTPException(status_code=400, detail="Symbol is required.")
        try:
            outcomes = symbol_outcomes(
                store,
                symbol,
                horizon_days=horizon or settings.calibration_horizon_days,
                bucket_width=bucket_width or settings.calibration_bucket_width,
                min_observations=min_observations or 5,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return outcomes_to_wire(outcomes)


__all__ = ["register_history_routes"]
