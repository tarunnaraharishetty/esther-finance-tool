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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

from src.config import Settings, get_settings
from src.data.envelope import DataEnvelope
from src.data.freshness import evaluate_freshness, policy_for
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
    ReportPeriod,
)
from src.intelligence.fundamentals.reconciliation import (
    FieldDivergence,
    ReconciliationWarning,
    reconcile,
)
from src.intelligence.fundamentals.sec_edgar import SecEdgarProvider
from src.intelligence.fundamentals.yahoo_fallback import YahooFallbackProvider
from src.utils.logging import get_logger

if TYPE_CHECKING:
    # Avoid circular import: health_store.py imports ProviderHealth from
    # this package's models. The service only needs HealthStore as a
    # type annotation — actual instances are duck-typed via record().
    from src.data.health_store import HealthStore
    from src.data.retry_queue import RetryQueue

log = get_logger(__name__)


# Provider error statuses that indicate "try again later might work".
# These are the only conditions under which we enqueue for retry.
# Mixed/permanent statuses (e.g. unavailable due to no API key) won't
# resolve themselves, so we don't pollute the queue with them.
_TRANSIENT_STATUSES = frozenset({"rate_limited", "transient"})


_STATUS_OK = "ok"
_STATUS_EMPTY = "empty"
_STATUS_RATE_LIMITED = "rate_limited"
_STATUS_UNAVAILABLE = "unavailable"
_STATUS_TRANSIENT = "transient"
_STATUS_SKIPPED = "skipped"


# Provider-confidence curve indexed by chain position of the successful
# provider. Position 0 = primary (FMP by default) → fully trusted. Each
# step down the chain ratchets confidence toward 0.4 — the floor for
# the last-resort scraped fallback (Yahoo). The curve is opinionated:
# a Yahoo answer is worth less than an FMP answer at the same freshness
# tier because the upstream is unauthenticated and lossy. Externalize
# to Settings if trader feedback warrants tuning.
_PROVIDER_CONFIDENCE_BY_POSITION: tuple[float, ...] = (
    1.00,
    0.85,
    0.70,
    0.55,
    0.40,
)


def _confidence_for_position(position: int) -> float:
    """Look up the confidence weight for a chain-position index.

    Positions past the table tail collapse to the last value (0.40).
    Negative positions are clamped to 1.0 — defensive against
    misconfiguration.
    """
    if position < 0:
        return _PROVIDER_CONFIDENCE_BY_POSITION[0]
    if position >= len(_PROVIDER_CONFIDENCE_BY_POSITION):
        return _PROVIDER_CONFIDENCE_BY_POSITION[-1]
    return _PROVIDER_CONFIDENCE_BY_POSITION[position]


def _ensure_utc(value: datetime) -> datetime:
    """Coerce a naive datetime to UTC; pass tz-aware through unchanged.

    Provider responses are inconsistent — FMP returns naive ISO dates,
    SEC EDGAR returns explicit timezones, others vary. The envelope's
    freshness math subtracts ``as_of`` from a UTC ``fetched_at`` so
    any naive value would raise. We assume naive == UTC, which matches
    every fundamentals provider's published convention.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _resolve_as_of(
    normalized: NormalizedFundamentals, fallback: datetime
) -> tuple[datetime, str]:
    """Pick the canonical ``as_of`` timestamp + matching policy key.

    Strategy:

    * If any quarterly statement is present, ``as_of`` = the most
      recent quarterly ``fiscal_date`` and the policy is
      ``fundamentals.quarterly``. Quarterly cadence is the more
      granular contract — when it's available, use it.
    * Else if any annual statement is present, ``as_of`` = the most
      recent annual ``fiscal_date`` and policy is
      ``fundamentals.annual``.
    * Else fall back to ``fallback - 1 day`` (the provider returned
      profile-only data) and the quarterly policy. The synthetic
      one-day offset prevents a profile-only record from looking
      pathologically "fresh"; the quarterly policy is the
      conservative default (faster expiry).
    """
    all_dates: list[tuple[datetime, ReportPeriod]] = []
    all_dates.extend((s.fiscal_date, s.period) for s in normalized.income_statements)
    all_dates.extend((s.fiscal_date, s.period) for s in normalized.balance_sheets)
    all_dates.extend((s.fiscal_date, s.period) for s in normalized.cash_flows)

    if not all_dates:
        return fallback - timedelta(days=1), "fundamentals.quarterly"

    quarterly = [d for d in all_dates if d[1] is ReportPeriod.QUARTERLY]
    if quarterly:
        return max(quarterly, key=lambda d: d[0])[0], "fundamentals.quarterly"

    annual = [d for d in all_dates if d[1] is ReportPeriod.ANNUAL]
    if annual:
        return max(annual, key=lambda d: d[0])[0], "fundamentals.annual"

    # Only TTM / other periods present — treat as quarterly cadence;
    # TTM rolls quarterly so the quarterly window is the right read.
    return max(all_dates, key=lambda d: d[0])[0], "fundamentals.quarterly"


@dataclass(frozen=True)
class FundamentalsResult:
    """What the orchestrator returns to its callers.

    Carries the freshness-grade :class:`DataEnvelope` wrapping the
    normalized payload, plus the raw provider response and the full
    per-provider health log for this call. The DB layer in Phase 5
    persists all three.

    When the primary chain position is > 0 and reconciliation is
    enabled, ``divergences`` carries per-field disagreements against
    a secondary provider's read of the same symbol. Empty tuple
    means either reconciliation didn't run (chain position 0, or
    disabled, or no secondary available) or the two providers agreed
    within the configured threshold. ``reconciliation_warnings``
    captures per-field notes about *why* a field wasn't compared
    (e.g., fiscal-date mismatch).

    The ``.fundamentals`` property is the most-used access path; it
    unwraps ``envelope.data`` so existing callsites read cleanly.
    """

    envelope: DataEnvelope[NormalizedFundamentals]
    raw: ProviderRawResponse
    health: tuple[ProviderHealth, ...] = field(default_factory=tuple)
    divergences: tuple[FieldDivergence, ...] = field(default_factory=tuple)
    reconciliation_warnings: tuple[ReconciliationWarning, ...] = field(
        default_factory=tuple
    )

    @property
    def fundamentals(self) -> NormalizedFundamentals:
        """Convenience accessor for ``envelope.data``."""
        return self.envelope.data


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
        health_store: HealthStore | None = None,
        retry_queue: RetryQueue | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = False
        self._health_store = health_store
        self._retry_queue = retry_queue
        if providers is not None:
            self._providers = providers
        else:
            self._providers = _build_default_chain(self._settings, self._client)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        health_store: HealthStore | None = None,
        retry_queue: RetryQueue | None = None,
    ) -> FundamentalsService:
        return cls(
            settings=settings,
            health_store=health_store,
            retry_queue=retry_queue,
        )

    def _sink_health(self, rows: list[ProviderHealth]) -> None:
        """Write per-call health rows to the store, swallowing failures.

        Sink writes are best-effort: the persistent health log exists
        so operators can see "FMP success rate dropped" — losing one
        batch of rows is acceptable; *breaking a fundamentals fetch
        because the DB is locked* is not. We catch broadly and log.
        """
        if self._health_store is None or not rows:
            return
        try:
            self._health_store.record(rows)
        except Exception as exc:
            log.warning(
                "health_store.record.failed",
                error=str(exc),
                row_count=len(rows),
            )

    async def _maybe_reconcile(
        self,
        symbol: str,
        primary_index: int,
        primary_normalized: NormalizedFundamentals,
    ) -> tuple[tuple[FieldDivergence, ...], tuple[ReconciliationWarning, ...]]:
        """Run one cross-provider reconciliation call.

        Fires only when:

        * :attr:`Settings.reconciliation_enabled` is True (operator
          can opt out where API spend is the binding constraint), AND
        * ``primary_index > 0`` — we only pay the extra call when we
          fell through to a fallback. The happy path stays single-call.

        Secondary provider selection: first configured provider in
        the chain *after* the primary. Earlier-position providers
        already failed this call, so re-asking them would waste a
        quota and almost certainly fail again.

        Returns empty tuples on any non-success — secondary not
        configured, secondary fetch failed, no comparable fields.
        Reconciliation never breaks the primary fetch.
        """
        if not self._settings.reconciliation_enabled:
            return (), ()
        if primary_index == 0:
            return (), ()

        secondary_provider: FundamentalsProvider | None = None
        for prov in self._providers[primary_index + 1 :]:
            if prov.is_configured:
                secondary_provider = prov
                break
        if secondary_provider is None:
            return (), ()

        try:
            secondary_normalized, _raw = await secondary_provider.fetch(symbol)
        except Exception as exc:
            # Reconciliation must never break the primary fetch. The
            # primary already succeeded; we'd rather ship the data
            # without divergence info than 503 the trader. Log so the
            # operator can see why divergences are absent.
            log.warning(
                "reconciliation.secondary_fetch.failed",
                symbol=symbol,
                secondary=secondary_provider.name.value,
                error=str(exc),
            )
            return (), ()

        result = reconcile(
            primary_normalized,
            secondary_normalized,
            threshold=self._settings.reconciliation_divergence_threshold,
        )
        log.info(
            "reconciliation.done",
            symbol=symbol,
            primary=primary_normalized.primary_provider.value,
            secondary=secondary_provider.name.value,
            divergence_count=len(result.divergences),
            warning_count=len(result.warnings),
        )
        return result.divergences, result.warnings

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

            fetched_at = datetime.now(UTC)
            raw_as_of, policy_key = _resolve_as_of(normalized, fetched_at)
            as_of = _ensure_utc(raw_as_of)
            freshness = evaluate_freshness(as_of, fetched_at, policy_for(policy_key))
            base_confidence = _confidence_for_position(index)

            # Cross-provider reconciliation on high-trust fields when we
            # fell through to a fallback. Returns empty tuples on the
            # happy path or any failure mode — never raises.
            divergences, recon_warnings = await self._maybe_reconcile(
                sym, index, normalized
            )
            confidence = _confidence_after_divergences(base_confidence, divergences)

            envelope: DataEnvelope[NormalizedFundamentals] = DataEnvelope(
                data=normalized,
                as_of=as_of,
                fetched_at=fetched_at,
                source_chain=tuple(p.value for p in attempted),
                freshness=freshness,
                provider_confidence=confidence,
            )

            # Persist the per-call health log for the rolling SLO view.
            # Best-effort: a sink failure must not break the fetch.
            self._sink_health(health)

            return FundamentalsResult(
                envelope=envelope,
                raw=raw,
                health=tuple(health),
                divergences=divergences,
                reconciliation_warnings=recon_warnings,
            )

        # Chain exhausted with nothing healthy. Sink the health log
        # *before* raising — the failure rows are the operator's
        # primary signal that the chain is degraded; losing them
        # would defeat the purpose of having a store.
        self._sink_health(health)
        # If every observed failure was transient/rate-limited the
        # symbol is a retry candidate — a 30-second blip shouldn't
        # strand it until the next user click. Permanent failures
        # (no API key, etc.) won't resolve themselves and don't queue.
        self._maybe_enqueue_for_retry(sym, health, chain_errors)
        # Raise an aggregate error so callers can decide whether to
        # 503, cache-fallback, or render a "no data" badge in the UI.
        raise ProviderChainExhausted(sym, tuple(health), tuple(chain_errors))

    def _maybe_enqueue_for_retry(
        self,
        symbol: str,
        health: list[ProviderHealth],
        chain_errors: list[str],
    ) -> None:
        """Enqueue ``symbol`` when the chain's observed failures are
        all transient or rate-limited.

        Skips:

        * No retry queue wired (operator opted out via settings).
        * Any chain failure was non-transient — wouldn't recover on retry.
        * No actual failure rows (defensive; should never fire on the
          chain-exhausted path).
        """
        if self._retry_queue is None:
            return
        observed_statuses = {
            h.status for h in health if h.status not in ("skipped", "ok")
        }
        if not observed_statuses:
            return
        if not observed_statuses.issubset(_TRANSIENT_STATUSES):
            return
        try:
            self._retry_queue.enqueue(symbol, tuple(chain_errors))
        except Exception as exc:
            # Enqueue is best-effort: the caller is about to raise
            # ProviderChainExhausted regardless, and we'd rather they
            # see the original error than a queue write failure.
            log.warning(
                "retry_queue.enqueue.failed",
                symbol=symbol,
                error=str(exc),
            )


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


def _confidence_after_divergences(
    confidence: float, divergences: tuple[FieldDivergence, ...]
) -> float:
    """Down-weight provider confidence in the presence of divergences.

    Multiplicative penalty of 0.85 per divergence, floored at 0.5x.
    A single divergence drops confidence by ~15%; three drop it to
    ~61%; beyond three the floor kicks in. The floor keeps the
    envelope from looking unusable when a long tail of small
    disagreements piles up.
    """
    if not divergences:
        return confidence
    penalty = 0.85 ** len(divergences)
    penalty = max(penalty, 0.5)
    return confidence * penalty


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
