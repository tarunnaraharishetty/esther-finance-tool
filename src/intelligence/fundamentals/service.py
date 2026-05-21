"""Fundamentals orchestrator.

Walks a configured provider chain (FMP -> Finnhub -> Alpha Vantage ->
SEC EDGAR -> Yahoo) and returns the first :class:`NormalizedFundamentals`
that comes back healthy. Skipped providers are recorded as
:class:`ProviderHealth` entries so callers can log / persist per-call
provider health and detect a silently-rotting upstream.

In Phase 1 the chain returns the first healthy provider's full payload
— we don't merge across providers yet. Phase 2 may layer in a
"fill missing fields from secondary provider" pass, but that adds
complexity (whose ratio do you trust when both disagree?) and isn't
needed to ship the analyzer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import httpx

from src.config import Settings, get_settings
from src.intelligence.fundamentals.alphavantage import AlphaVantageProvider
from src.intelligence.fundamentals.base import (
    FundamentalsProvider,
    ProviderError,
    ProviderNotFound,
    ProviderRateLimited,
    ProviderTransient,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.finnhub import FinnhubProvider
from src.intelligence.fundamentals.fmp import FmpProvider
from src.intelligence.fundamentals.models import (
    NormalizedFundamentals,
    ProviderHealth,
    ProviderName,
    ProviderRawResponse,
)
from src.intelligence.fundamentals.sec_edgar import SecEdgarProvider
from src.intelligence.fundamentals.yahoo_fallback import YahooFallbackProvider
from src.utils.logging import get_logger

log = get_logger(__name__)


_STATUS_OK = "ok"
_STATUS_EMPTY = "empty"
_STATUS_RATE_LIMITED = "rate_limited"
_STATUS_UNAVAILABLE = "unavailable"
_STATUS_TRANSIENT = "transient"
_STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class FundamentalsResult:
    """What the orchestrator returns to its callers.

    Carries the normalized payload + the raw response that produced it
    + the full per-provider health log for this call. The DB layer in
    Phase 5 persists all three.
    """

    fundamentals: NormalizedFundamentals
    raw: ProviderRawResponse
    health: tuple[ProviderHealth, ...] = field(default_factory=tuple)


class FundamentalsService:
    """Provider-chain orchestrator.

    Default chain order is FMP -> Finnhub -> Alpha Vantage -> SEC EDGAR
    -> Yahoo. Override by passing ``providers=...`` (tests) or
    ``Settings.fundamentals_provider_order`` (config).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        providers: list[FundamentalsProvider] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = False
        if providers is not None:
            self._providers = providers
        else:
            self._providers = _build_default_chain(self._settings, self._client)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "FundamentalsService":
        return cls(settings=settings)

    async def fetch(self, symbol: str) -> FundamentalsResult:
        """Walk the chain; return the first healthy provider's result.

        Each provider's outcome (ok / empty / rate_limited / unavailable
        / transient / skipped) becomes a :class:`ProviderHealth` row.
        After a provider returns ``ok`` the rest of the chain is marked
        ``skipped`` so the health log is complete for downstream
        analytics (e.g. "how often does FMP succeed?").
        """
        sym = symbol.upper()
        health: list[ProviderHealth] = []
        chain_errors: list[str] = []

        for index, provider in enumerate(self._providers):
            if not provider.is_configured:
                health.append(
                    _health_row(
                        provider.name,
                        sym,
                        _STATUS_UNAVAILABLE,
                        latency_ms=0.0,
                        error="provider not configured",
                    )
                )
                continue

            start = time.monotonic()
            try:
                normalized, raw = await provider.fetch(sym)
            except ProviderRateLimited as exc:
                latency_ms = (time.monotonic() - start) * 1000.0
                health.append(
                    _health_row(
                        provider.name, sym, _STATUS_RATE_LIMITED, latency_ms, str(exc)
                    )
                )
                chain_errors.append(f"{provider.name.value}: rate-limited")
                log.warning(
                    "fundamentals.provider.rate_limited",
                    provider=provider.name.value,
                    symbol=sym,
                )
                continue
            except ProviderNotFound as exc:
                latency_ms = (time.monotonic() - start) * 1000.0
                health.append(
                    _health_row(provider.name, sym, _STATUS_EMPTY, latency_ms, str(exc))
                )
                chain_errors.append(f"{provider.name.value}: not-found")
                log.info(
                    "fundamentals.provider.empty",
                    provider=provider.name.value,
                    symbol=sym,
                )
                continue
            except ProviderUnavailable as exc:
                latency_ms = (time.monotonic() - start) * 1000.0
                health.append(
                    _health_row(
                        provider.name, sym, _STATUS_UNAVAILABLE, latency_ms, str(exc)
                    )
                )
                chain_errors.append(f"{provider.name.value}: unavailable")
                log.warning(
                    "fundamentals.provider.unavailable",
                    provider=provider.name.value,
                    symbol=sym,
                    error=str(exc),
                )
                continue
            except ProviderTransient as exc:
                latency_ms = (time.monotonic() - start) * 1000.0
                health.append(
                    _health_row(
                        provider.name, sym, _STATUS_TRANSIENT, latency_ms, str(exc)
                    )
                )
                chain_errors.append(f"{provider.name.value}: transient")
                log.warning(
                    "fundamentals.provider.transient",
                    provider=provider.name.value,
                    symbol=sym,
                    error=str(exc),
                )
                continue

            latency_ms = (time.monotonic() - start) * 1000.0
            health.append(
                _health_row(provider.name, sym, _STATUS_OK, latency_ms, None)
            )
            # Mark the remaining providers in the chain as skipped so
            # the health log reflects the full chain shape, not just
            # the ones tried.
            for skipped in self._providers[index + 1 :]:
                health.append(
                    _health_row(skipped.name, sym, _STATUS_SKIPPED, 0.0, None)
                )
            log.info(
                "fundamentals.fetch.ok",
                provider=provider.name.value,
                symbol=sym,
                latency_ms=int(latency_ms),
                chain_errors=chain_errors,
            )
            # Stamp the chain shape onto the normalized record so
            # downstream code knows who else was attempted.
            attempted = tuple(h.provider for h in health if h.status != _STATUS_SKIPPED)
            normalized = replace(normalized, contributing_providers=attempted)
            return FundamentalsResult(
                fundamentals=normalized, raw=raw, health=tuple(health)
            )

        # Chain exhausted with nothing healthy. Raise an aggregate
        # error so callers can decide whether to 503, cache-fallback,
        # or render a "no data" badge in the UI.
        raise ProviderChainExhausted(sym, tuple(health), tuple(chain_errors))


class ProviderChainExhausted(RuntimeError):
    """Every provider in the chain failed or was unconfigured."""

    def __init__(
        self,
        symbol: str,
        health: tuple[ProviderHealth, ...],
        errors: tuple[str, ...],
    ) -> None:
        joined = "; ".join(errors) if errors else "no providers configured"
        super().__init__(
            f"fundamentals chain exhausted for {symbol}: {joined}"
        )
        self.symbol = symbol
        self.health = health
        self.errors = errors


def _health_row(
    provider: ProviderName,
    symbol: str,
    status: str,
    latency_ms: float,
    error: str | None,
) -> ProviderHealth:
    return ProviderHealth(
        provider=provider,
        symbol=symbol,
        status=status,
        latency_ms=latency_ms,
        checked_at=datetime.now(UTC),
        error_message=error,
    )


def _build_default_chain(
    settings: Settings, client: httpx.AsyncClient | None
) -> list[FundamentalsProvider]:
    """Build the ordered provider chain from settings.

    ``fundamentals_provider_order`` is a list of :class:`ProviderName`
    values; unknown entries fail fast at settings load (validator
    rejects them) so by the time we get here every name is valid.
    """
    by_name: dict[ProviderName, FundamentalsProvider] = {
        ProviderName.FMP: FmpProvider(settings, client=client),
        ProviderName.FINNHUB: FinnhubProvider(settings, client=client),
        ProviderName.ALPHA_VANTAGE: AlphaVantageProvider(settings, client=client),
        ProviderName.SEC_EDGAR: SecEdgarProvider(settings, client=client),
        ProviderName.YAHOO: YahooFallbackProvider(settings, client=client),
    }
    chain: list[FundamentalsProvider] = []
    for raw_name in settings.fundamentals_provider_order:
        try:
            name = ProviderName(raw_name)
        except ValueError:
            # The settings validator already rejects unknown names, so
            # in practice this branch is unreachable. Belt-and-braces
            # for the case where chain order is constructed manually.
            continue
        provider = by_name.get(name)
        if provider is not None:
            chain.append(provider)
    return chain


__all__ = [
    "FundamentalsResult",
    "FundamentalsService",
    "ProviderChainExhausted",
    "ProviderError",
]
