"""Provider protocol + error taxonomy.

Every concrete provider implements :class:`FundamentalsProvider`. The
orchestrator
(:class:`~src.intelligence.fundamentals.service.FundamentalsService`)
only depends on this surface so swapping providers in tests is a
one-line ``providers=[FakeProvider()]`` substitution.

Errors are split into four categories so the orchestrator can decide
whether to fall back to the next provider or surface to the caller:

* :class:`ProviderUnavailable` — the provider is unusable for this
  call (no API key, hard 4xx auth failure, persistent 5xx). Skip to
  the next provider, log health as ``unavailable``.
* :class:`ProviderRateLimited` — quota exhausted. Skip to the next
  provider, log health as ``rate_limited``. Backoff happens inside
  the provider's HTTP layer; the orchestrator does not retry the
  same provider in-call.
* :class:`ProviderNotFound` — the provider responded fine but has no
  data for this symbol. Skip to the next provider, log ``empty``. Do
  not treat this as a failure of the chain — it's expected for OTC
  / international / freshly-listed names.
* :class:`ProviderTransient` — short network blip, malformed JSON
  blob, etc. Skip to the next provider, log ``transient``.

Anything else (e.g. a programming bug in the mapper) propagates as a
normal exception so it surfaces in tests and logs loudly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.intelligence.fundamentals.models import (
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
)


class ProviderError(Exception):
    """Base class for orchestrator-handled provider failures."""

    def __init__(self, provider: ProviderName, message: str) -> None:
        super().__init__(f"[{provider.value}] {message}")
        self.provider = provider
        self.message = message


class ProviderUnavailable(ProviderError):
    """Provider cannot serve this call (missing key, auth fail, 5xx)."""


class ProviderRateLimited(ProviderError):
    """Provider returned a 429 / quota-exhausted response.

    ``retry_after_seconds`` carries the ``Retry-After`` header value when
    the upstream supplied one (per RFC 6585). ``None`` when the response
    had no header or the value couldn't be parsed. Honoured by the retry
    queue scheduler so a provider asking for a 5-minute wait isn't
    retried 30 seconds later by our default backoff.
    """

    def __init__(
        self,
        provider: ProviderName,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(provider, message)
        self.retry_after_seconds = retry_after_seconds


class ProviderNotFound(ProviderError):
    """Provider has no data for this symbol (200 OK + empty payload)."""


class ProviderTransient(ProviderError):
    """Transient network / parsing error — try next provider."""


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Surface every fundamentals provider must implement.

    Implementations should:

    * Raise one of the four :class:`ProviderError` subclasses for any
      orchestrator-handled failure. Programming bugs may surface as
      normal exceptions.
    * Return a fully populated :class:`NormalizedFundamentals` keyed
      to this provider when ``fetch`` succeeds.
    * Also return the matching :class:`ProviderRawResponse` so the
      orchestrator can persist the raw payload alongside the
      normalized one (the analyzer DB schema in Phase 5 stores both).
    """

    name: ProviderName

    @property
    def is_configured(self) -> bool:
        """Return True iff this provider has the credentials it needs.

        The orchestrator skips unconfigured providers without raising
        so a missing API key is a silent no-op, not a startup failure.
        Each provider decides what "configured" means — most rely on
        an API key in settings, SEC EDGAR only needs a User-Agent
        string, Yahoo needs nothing.
        """
        ...

    async def fetch(
        self, symbol: str
    ) -> tuple[NormalizedFundamentals, ProviderRawResponse]:
        """Fetch + normalize fundamentals for ``symbol``.

        Returns a tuple of (normalized, raw). The orchestrator persists
        the raw payload for replay / re-parse without re-fetching.
        """
        ...


__all__ = [
    "FundamentalsProvider",
    "ProviderError",
    "ProviderNotFound",
    "ProviderRateLimited",
    "ProviderTransient",
    "ProviderUnavailable",
]
