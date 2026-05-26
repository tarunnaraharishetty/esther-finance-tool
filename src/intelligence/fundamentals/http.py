"""Shared HTTP helper for fundamentals providers.

All concrete providers share the same failure-classification logic:
4xx auth -> ``ProviderUnavailable``, 429 -> ``ProviderRateLimited``,
5xx / network -> ``ProviderTransient``, 200 OK with empty body ->
``ProviderNotFound``. Centralizing the mapping here keeps each
provider focused on its endpoint URLs + response shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from src.intelligence.fundamentals.base import (
    ProviderNotFound,
    ProviderRateLimited,
    ProviderTransient,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.models import ProviderName

_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header value into seconds.

    RFC 7231 allows two forms: a non-negative integer (delta-seconds) or
    an HTTP-date (absolute deadline). Returns ``None`` when the value
    is missing, malformed, or in the past. We cap at 24h so a hostile
    or buggy upstream can't strand a symbol for weeks.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    # Try delta-seconds first — by far the common form.
    try:
        delta = float(stripped)
    except ValueError:
        delta = None
    if delta is not None:
        if delta < 0:
            return None
        return min(delta, 24 * 3600.0)
    # Fall back to HTTP-date.
    try:
        deadline = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if deadline is None:
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    delta = (deadline - datetime.now(UTC)).total_seconds()
    if delta <= 0:
        return None
    return min(delta, 24 * 3600.0)


async def get_json(
    provider: ProviderName,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: httpx.Timeout = _DEFAULT_TIMEOUT,  # noqa: ASYNC109 — forwarded to httpx, not a sleep deadline
) -> Any:
    """Issue a GET, return parsed JSON, raise typed ProviderErrors.

    The orchestrator catches the typed errors and decides whether to
    fall back. Anything else (programming bug, JSON-parse failure on
    what should be JSON) bubbles up so loud failures stay loud.

    ``client`` is accepted so the service can share one ``httpx.AsyncClient``
    across providers — cheaper than spinning a fresh one per call.
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderTransient(provider, f"network error: {exc}") from exc

        status = response.status_code
        if status == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise ProviderRateLimited(
                provider,
                "rate limit hit (HTTP 429)",
                retry_after_seconds=retry_after,
            )
        if status in (401, 403):
            raise ProviderUnavailable(provider, f"auth failure (HTTP {status})")
        if status == 404:
            raise ProviderNotFound(provider, "no data for symbol (HTTP 404)")
        if 500 <= status < 600:
            raise ProviderTransient(provider, f"upstream error (HTTP {status})")
        if status >= 400:
            raise ProviderUnavailable(provider, f"client error (HTTP {status})")

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderTransient(provider, f"non-JSON response: {exc}") from exc

        if _is_effectively_empty(data):
            raise ProviderNotFound(provider, "empty payload")
        return data
    finally:
        if owns_client:
            await client.aclose()


def _is_effectively_empty(payload: Any) -> bool:
    """Treat ``[]``, ``{}``, ``None`` as not-found.

    Many providers respond with an empty array rather than 404 when a
    symbol is unknown. The orchestrator wants those treated the same
    way as a real 404 so the fallback chain advances.
    """
    if payload is None:
        return True
    return isinstance(payload, (list, dict)) and len(payload) == 0


__all__ = ["get_json"]
