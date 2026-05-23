"""Provider-health observability API.

Surfaces the rolling SLO view from :class:`HealthStore`, the empirical
accuracy ledger from :class:`AccuracyStore`, the composed trust weight
from :class:`ProviderTrust`, and the in-flight :class:`RetryQueue`
state — so operators (and the trader-facing Providers page) can answer
"why is FMP down-weighted right now?" without SSH-ing into SQLite.

Routes
------
* ``GET /api/health/providers`` — per-provider rolling summary + trust
  breakdown + per-field accuracy + recent failure rows. Accepts a
  ``window=<int><s|m|h|d>`` query param (default ``1h``); rejects
  anything else with 400.
* ``GET /api/health/queue`` — :class:`RetryQueue` snapshot.

Union of provider sources
-------------------------
The endpoint emits a row for every provider that appears in *either*
the health log OR the accuracy ledger over the window. A provider
with only accuracy events (no health rows yet because nothing failed)
would be invisible under the legacy "health-store only" enumeration —
the Trust Score wouldn't have a place to surface its contribution.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from src.data.health_store import HealthStore, ProviderSummary
from src.data.retry_queue import RetryQueue, RetryStatus
from src.intelligence.fundamentals.models import ProviderName
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.accuracy_store import AccuracyStore
    from src.intelligence.fundamentals.provider_trust import ProviderTrust

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
    accuracy_store: AccuracyStore | None = None,
    provider_trust: ProviderTrust | None = None,
) -> None:
    """Attach provider-health routes to ``app``.

    Args:
        app: FastAPI app to extend.
        store: Bound :class:`HealthStore`. Closes over it so the same
            instance backs every call.
        retry_queue: Optional bound :class:`RetryQueue`. When provided,
            registers the ``/api/health/queue`` endpoint.
        accuracy_store: Optional bound :class:`AccuracyStore`. When
            provided, per-provider rows carry a ``field_accuracy``
            block (rolling per-high-trust-field agree rates).
        provider_trust: Optional bound :class:`ProviderTrust`. When
            provided, per-provider rows carry a ``trust_breakdown``
            block (the composite weight + its components). The
            Providers page keys off this to render the trust gauge.
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
        """Return rolling per-provider health + trust + recent failures.

        On a fresh store with no data, returns an empty ``providers``
        array and empty ``recent_failures``. Never 500s on "no data" —
        that's expected on a first boot.
        """
        try:
            window_td = _parse_window(window)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        moment = datetime.now(UTC)
        summaries = store.summarize(window_td, now=moment)
        failures = store.list_recent_failures(limit=recent_limit)

        # Index health summaries by provider name for the union step.
        summary_by_name: dict[str, ProviderSummary] = {
            s.provider: s for s in summaries
        }

        # Union: every provider that has either a health row OR an
        # accuracy event in the window. Providers seen only in the
        # accuracy ledger get a degraded "no health rows" summary so
        # the Trust Score's contribution still surfaces.
        accuracy_provider_names: tuple[str, ...] = ()
        if accuracy_store is not None:
            try:
                accuracy_provider_names = accuracy_store.known_providers()
            except Exception as exc:
                # Best-effort: a stuck accuracy store must not break
                # the health endpoint. Log and degrade.
                log.warning(
                    "health.accuracy_store.enumerate_failed",
                    error=str(exc),
                )

        all_names = sorted(
            set(summary_by_name.keys()) | set(accuracy_provider_names)
        )

        provider_rows: list[dict[str, Any]] = []
        for name in all_names:
            summary = summary_by_name.get(name)
            if summary is None:
                summary = _degraded_summary(name, window_td)
            row = _summary_to_wire(summary)
            row["trust_breakdown"] = _trust_block(provider_trust, name)
            row["field_accuracy"] = _field_accuracy_block(
                accuracy_store, name, window_td
            )
            provider_rows.append(row)

        return {
            "as_of": moment.isoformat(),
            "window_seconds": int(window_td.total_seconds()),
            "providers": provider_rows,
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


# ---------------------------------------------------------------------------
# Composer helpers
# ---------------------------------------------------------------------------


def _trust_block(
    provider_trust: ProviderTrust | None, provider: str
) -> dict[str, Any] | None:
    """Resolve the :class:`TrustBreakdown` for ``provider``, or None.

    ``None`` only when provider_trust isn't wired or the breakdown
    call raises — the frontend reads ``trust_breakdown is null`` as
    "trust ledger not configured" and renders a placeholder.
    """
    if provider_trust is None:
        return None
    try:
        pname = ProviderName(provider)
    except ValueError:
        # Provider name unknown to the enum (deleted provider, etc.) —
        # skip the trust lookup rather than crashing the row.
        return None
    try:
        breakdown = provider_trust.breakdown_for(pname)
    except Exception as exc:
        log.warning(
            "health.trust_breakdown.failed",
            provider=provider,
            error=str(exc),
        )
        return None
    return breakdown.to_dict()


def _field_accuracy_block(
    accuracy_store: AccuracyStore | None,
    provider: str,
    window: timedelta,
) -> dict[str, Any] | None:
    """Build the per-field accuracy summary for ``provider``.

    Returns ``None`` when the accuracy store isn't wired or the
    lookup raises. Empty ``by_field`` (no events in window) returns
    a populated summary with zero counts so the UI can render
    "no observations yet" honestly.
    """
    if accuracy_store is None:
        return None
    try:
        summary = accuracy_store.summarize(provider, window)
    except Exception as exc:
        log.warning(
            "health.field_accuracy.failed",
            provider=provider,
            error=str(exc),
        )
        return None
    return summary.to_dict()


def _degraded_summary(provider: str, window: timedelta) -> ProviderSummary:
    """Build a zero-count ProviderSummary for a provider that's only in
    the accuracy ledger (no health rows yet).

    Lets the union step include the provider in the response without
    branching the wire shape per source. ``last_seen_at`` /
    ``last_error_at`` are ``None`` so the UI shows "no health calls
    yet" rather than a fake timestamp.
    """
    return ProviderSummary(provider=provider, window=window, total=0)


# ---------------------------------------------------------------------------
# Per-record wire serializers
# ---------------------------------------------------------------------------


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


def _summary_to_wire(summary: ProviderSummary) -> dict[str, Any]:
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
