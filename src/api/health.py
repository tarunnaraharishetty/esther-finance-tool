"""Provider-health observability API.

Surfaces the rolling SLO view from :class:`HealthStore` and the in-flight
:class:`RetryQueue` state so operators and the trader-facing UI can
see "Fundamentals provider FMP degraded · 76% success in the last
hour · 3 symbols queued for retry" instead of an unexplained spinner.

Routes
------
* ``GET /api/health/providers`` — rolling summary per provider, plus
  the 20 most recent failure rows. Accepts a ``window=<int><s|m|h|d>``
  query param (default ``1h``); rejects anything else with 400.
* ``GET /api/health/queue`` — :class:`RetryQueue` snapshot: counts of
  pending vs permanently-failed entries plus the full entry list.

Why one endpoint instead of two
-------------------------------
Trader-facing UI surfaces the rolling summary + recent-failures list
together (it's the same "what is FMP doing right now" question). One
fetch + one render is simpler than two coordinated calls.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from src.data.health_store import HealthStore
from src.data.retry_queue import RetryQueue, RetryStatus
from src.utils.logging import get_logger

log = get_logger(__name__)


# Cap the window at 30d so a runaway query can't pull months of rows.
# Anything longer should be a separate batch job, not a synchronous API.
_MAX_WINDOW = timedelta(days=30)

_WINDOW_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")
_UNIT_TO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_window(value: str) -> timedelta:
    """Translate a ``\\d+[smhd]`` token into a positive timedelta.

    Raises :class:`ValueError` on a malformed token or one that
    exceeds :data:`_MAX_WINDOW`. The route catches and translates to
    HTTP 400 so the user sees a readable error rather than a 500.
    """
    match = _WINDOW_PATTERN.match(value)
    if match is None:
        raise ValueError(
            f"invalid window {value!r}; expected <integer><s|m|h|d> "
            "(e.g. '15m', '1h', '24h', '7d')"
        )
    qty = int(match.group("value"))
    if qty <= 0:
        raise ValueError(f"window must be positive, got {value!r}")
    seconds = qty * _UNIT_TO_SECONDS[match.group("unit")]
    window = timedelta(seconds=seconds)
    if window > _MAX_WINDOW:
        raise ValueError(
            f"window {value!r} exceeds maximum of {_MAX_WINDOW}"
        )
    return window


def register_health_routes(
    app: FastAPI,
    *,
    store: HealthStore,
    retry_queue: RetryQueue | None = None,
) -> None:
    """Attach provider-health routes to ``app``.

    Args:
        app: FastAPI app to extend.
        store: Bound :class:`HealthStore`. The route closes over it so
            the same instance backs every call. Tests inject a
            ``tmp_path``-rooted store; production gets the default
            from ``Settings.health_store_path``.
        retry_queue: Optional bound :class:`RetryQueue`. When provided,
            registers the ``/api/health/queue`` endpoint. ``None``
            means the operator opted out of retry-queue persistence
            (``RETRY_QUEUE_PATH=``) and the queue endpoint 404s.
    """

    @app.get("/api/health/providers")
    async def providers(
        window: str = Query(
            default="1h",
            description=(
                "Rolling window. Accepts <int><s|m|h|d> (e.g. '15m', "
                "'1h', '24h'). Capped at 30d."
            ),
        ),
        recent_limit: int = Query(
            default=20,
            ge=0,
            le=200,
            description="Max recent failure rows to include.",
        ),
    ) -> dict[str, Any]:
        """Return rolling per-provider health + recent failures.

        On a fresh store with no data, returns an empty
        ``providers`` array and empty ``recent_failures``. The endpoint
        never 500s on "no data" — that's expected on a first boot.
        """
        try:
            window_td = _parse_window(window)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        moment = datetime.now(UTC)
        summaries = store.summarize(window_td, now=moment)
        failures = store.list_recent_failures(limit=recent_limit)
        return {
            "as_of": moment.isoformat(),
            "window_seconds": int(window_td.total_seconds()),
            "providers": [_summary_to_wire(s) for s in summaries],
            "recent_failures": jsonable_encoder([asdict(f) for f in failures]),
        }

    if retry_queue is not None:

        @app.get("/api/health/queue")
        async def queue() -> dict[str, Any]:
            """Return the retry-queue snapshot.

            On a fresh queue with no data, returns empty counts and an
            empty ``entries`` array. Never 500s on "no data".
            """
            entries = retry_queue.all_entries()
            pending = sum(1 for e in entries if e.status is RetryStatus.PENDING)
            permanently_failed = sum(
                1 for e in entries if e.status is RetryStatus.PERMANENTLY_FAILED
            )
            return {
                "as_of": datetime.now(UTC).isoformat(),
                "pending": pending,
                "permanently_failed": permanently_failed,
                "total": len(entries),
                "entries": [_entry_to_wire(e) for e in entries],
            }


def _entry_to_wire(entry: Any) -> dict[str, Any]:
    """Serialize a :class:`RetryEntry` to JSON-safe shape."""
    return {
        "symbol": entry.symbol,
        "enqueued_at": entry.enqueued_at.isoformat(),
        "next_attempt_at": entry.next_attempt_at.isoformat(),
        "attempts": entry.attempts,
        "last_errors": list(entry.last_errors),
        "status": entry.status.value,
        "last_error_at": (
            entry.last_error_at.isoformat()
            if entry.last_error_at is not None
            else None
        ),
    }


def _summary_to_wire(summary: Any) -> dict[str, Any]:
    """Serialize :class:`ProviderSummary` for the JSON wire shape.

    ``window`` is a ``timedelta`` on the dataclass; the wire form is
    integer seconds for clean cross-language consumption. ``breakdown``
    is included as a flat dict so the UI can render the per-status
    chips without recomputing.
    """
    return {
        "provider": summary.provider,
        "window_seconds": int(summary.window.total_seconds()),
        "total": summary.total,
        "ok": summary.ok,
        "rate_limited": summary.rate_limited,
        "unavailable": summary.unavailable,
        "transient": summary.transient,
        "empty": summary.empty,
        "skipped": summary.skipped,
        "success_rate": round(summary.success_rate, 4),
        "p50_latency_ms": (
            round(summary.p50_latency_ms, 2)
            if summary.p50_latency_ms is not None
            else None
        ),
        "p95_latency_ms": (
            round(summary.p95_latency_ms, 2)
            if summary.p95_latency_ms is not None
            else None
        ),
        "last_seen_at": (
            summary.last_seen_at.isoformat()
            if summary.last_seen_at is not None
            else None
        ),
        "last_error_at": (
            summary.last_error_at.isoformat()
            if summary.last_error_at is not None
            else None
        ),
        "breakdown": dict(summary.breakdown),
    }


__all__ = ["register_health_routes"]
