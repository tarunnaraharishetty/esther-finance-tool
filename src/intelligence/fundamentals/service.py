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
    AccuracyEvent,
    NormalizedFundamentals,
    ProviderHealth,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)
from src.intelligence.fundamentals.provider_trust import ProviderTrust
from src.intelligence.fundamentals.reconciliation import (
    FieldDivergence,
    ReconciliationWarning,
    reconcile,
)
from src.intelligence.fundamentals.sanity import sanitize_normalized
from src.intelligence.fundamentals.sec_edgar import SecEdgarProvider
from src.intelligence.fundamentals.yahoo_fallback import YahooFallbackProvider
from src.utils.logging import get_logger

if TYPE_CHECKING:
    # Two deferral reasons stacked here, both runtime-cycle avoidance:
    # * health_store.py imports ProviderHealth from this package's
    #   models, so the service only takes HealthStore as a type
    #   annotation and duck-types instances via record().
    # * SectorCohort transitively imports NormalizedFundamentals from
    #   THIS package's models — loading service.py via
    #   fundamentals/__init__.py would trip a circular import. Same
    #   duck-type pattern; runtime only calls ``cohort.observe(...)``.
    from src.data.accuracy_store import AccuracyStore
    from src.data.health_store import HealthStore
    from src.data.retry_queue import RetryQueue
    from src.intelligence.analyzer.sector_medians import SectorCohort

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
        accuracy_store: AccuracyStore | None = None,
        provider_trust: ProviderTrust | None = None,
        sector_cohort: SectorCohort | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = False
        self._health_store = health_store
        self._retry_queue = retry_queue
        self._accuracy_store = accuracy_store
        self._sector_cohort = sector_cohort
        # When the caller didn't pass an explicit trust computer, build
        # one from whatever stores we have. The result is cheap to
        # construct and gracefully no-ops when both stores are None
        # (returns 1.0 for every provider — neutral cold start).
        self._provider_trust = provider_trust or ProviderTrust(
            accuracy_store=accuracy_store,
            health_store=health_store,
        )
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
        accuracy_store: AccuracyStore | None = None,
        sector_cohort: SectorCohort | None = None,
    ) -> FundamentalsService:
        return cls(
            settings=settings,
            health_store=health_store,
            retry_queue=retry_queue,
            accuracy_store=accuracy_store,
            sector_cohort=sector_cohort,
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

    def _sink_cohort(self, record: NormalizedFundamentals) -> None:
        """Feed ``record`` into the sector-medians cohort, if wired.

        Best-effort: a failure here only affects the next analyzer's
        view of computed sector medians (the static seed remains as a
        fallback). It must never break a fundamentals fetch. Closes
        B-13 — without this hook the entire computed-medians pipeline
        sits dormant and lookup() always serves the static TOML.
        """
        if self._sector_cohort is None:
            return
        try:
            self._sector_cohort.observe(record)
        except Exception as exc:
            log.warning(
                "sector_cohort.observe_failed",
                symbol=record.symbol,
                error=str(exc),
            )

    def _sink_accuracy(self, events: tuple[AccuracyEvent, ...]) -> None:
        """Write accuracy ledger entries, swallowing failures.

        Same best-effort contract as :meth:`_sink_health`: a stuck DB
        write must never break a fundamentals fetch. The trust-weight
        cache is invalidated after a successful write so the next
        lookup picks up the new evidence within the TTL window.
        """
        if self._accuracy_store is None or not events:
            return
        try:
            self._accuracy_store.record(events)
        except Exception as exc:
            log.warning(
                "accuracy_store.record.failed",
                error=str(exc),
                event_count=len(events),
            )
            return
        # New evidence — drop cached weights so the next fetch sees
        # the freshest aggregate.
        self._provider_trust.invalidate()

    async def _maybe_reconcile(
        self,
        symbol: str,
        primary_index: int,
        primary_normalized: NormalizedFundamentals,
    ) -> tuple[
        tuple[FieldDivergence, ...],
        tuple[ReconciliationWarning, ...],
        tuple[AccuracyEvent, ...],
    ]:
        """Run one cross-provider reconciliation call.

        Fires only when:

        * :attr:`Settings.reconciliation_enabled` is True (operator
          can opt out where API spend is the binding constraint), AND
        * ``primary_index > 0`` — we only pay the extra call when we
          fell through to a fallback. The happy path stays single-call.

        Secondary provider selection: prefer SEC EDGAR when it's
        configured and isn't the primary itself — official filings
        are the strongest available reference. Otherwise fall back to
        the first configured provider after the primary. Earlier-
        position providers already failed this call, so re-asking
        them would waste a quota and almost certainly fail again.

        Returns three empty tuples on any non-success — secondary not
        configured, secondary fetch failed, no comparable fields.
        Reconciliation never breaks the primary fetch.
        """
        if not self._settings.reconciliation_enabled:
            return (), (), ()
        if primary_index == 0:
            return (), (), ()

        secondary_provider = self._select_reference_provider(
            primary_index, primary_normalized.primary_provider
        )
        if secondary_provider is None:
            return (), (), ()

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
            return (), (), ()

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
            event_count=len(result.events),
        )
        return result.divergences, result.warnings, result.events

    def _select_reference_provider(
        self, primary_index: int, primary_name: ProviderName
    ) -> FundamentalsProvider | None:
        """Pick the secondary used as a reference for reconciliation.

        Prefers SEC EDGAR when it's configured and isn't the primary
        we already used — official filings outrank any commercial
        provider as an accuracy reference. Falls back to the first
        configured provider after the primary in the chain when EDGAR
        isn't available.
        """
        edgar = next(
            (
                p
                for p in self._providers
                if p.name is ProviderName.SEC_EDGAR
                and p.name is not primary_name
                and p.is_configured
            ),
            None,
        )
        if edgar is not None:
            return edgar
        return next(
            (
                p
                for p in self._providers[primary_index + 1 :]
                if p.is_configured
            ),
            None,
        )

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
        # Maximum ``Retry-After`` value any rate-limited provider asked
        # us to wait. Honoured when the symbol lands in the retry queue
        # so a 5-minute server-side cooldown isn't ignored by our
        # default 30-second backoff.
        max_retry_after_seconds: float = 0.0

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
                if exc.retry_after_seconds and exc.retry_after_seconds > max_retry_after_seconds:
                    max_retry_after_seconds = exc.retry_after_seconds
                log.warning(
                    "fundamentals.provider.rate_limited",
                    provider=provider.name.value,
                    symbol=sym,
                    retry_after_seconds=exc.retry_after_seconds,
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

            # Sanity-bound provider numbers before the analyzer sees them.
            # Bad scalars (negative market cap, EPS > 5000, P/E > 10000,
            # inverted analyst high/low) become None and a warning is
            # attached to the record — same "missing data" path every
            # downstream model already handles, instead of a poisoned
            # input flipping a recommendation.
            normalized, sanitized_rejections = sanitize_normalized(normalized)
            if sanitized_rejections:
                log.warning(
                    "fundamentals.sanity.rejected",
                    provider=provider.name.value,
                    symbol=sym,
                    rejected_fields=[field for field, _ in sanitized_rejections],
                )

            fetched_at = datetime.now(UTC)
            raw_as_of, policy_key = _resolve_as_of(normalized, fetched_at)
            as_of = _ensure_utc(raw_as_of)
            freshness = evaluate_freshness(as_of, fetched_at, policy_for(policy_key))
            base_confidence = _confidence_for_position(index)

            # Cross-provider reconciliation on high-trust fields when we
            # fell through to a fallback. Returns empty tuples on the
            # happy path or any failure mode — never raises.
            divergences, recon_warnings, accuracy_events = await self._maybe_reconcile(
                sym, index, normalized
            )
            # Sanity rejections become self-comparison accuracy events so
            # a provider that consistently ships bogus values degrades
            # its trust weight — the same feedback loop that punishes
            # cross-provider divergences. Sentinel shape:
            # ``reference_provider = provider`` (self), ``rel_error = 1.0``,
            # ``agreed = False``, ``field`` namespaced with a ``sanity:``
            # prefix so the cross-provider aggregations stay distinct.
            sanity_events = _rejections_to_accuracy_events(
                provider.name, sym, sanitized_rejections, fetched_at
            )
            accuracy_events = accuracy_events + sanity_events
            confidence_after_divergences = _confidence_after_divergences(
                base_confidence, divergences
            )
            # Multiply by the empirical trust weight derived from
            # accuracy ledger + health log. Cold-start providers
            # return 1.0; the multiplier only bites once we have
            # statistically meaningful history. This is the entry
            # point through which the platform's "providers learn
            # from outcomes" moat enters every report.
            trust_weight = self._provider_trust.weight_for(provider.name)
            confidence = max(0.0, min(1.0, confidence_after_divergences * trust_weight))

            envelope: DataEnvelope[NormalizedFundamentals] = DataEnvelope(
                data=normalized,
                as_of=as_of,
                fetched_at=fetched_at,
                source_chain=tuple(p.value for p in attempted),
                freshness=freshness,
                provider_confidence=confidence,
            )

            # Persist the per-call health log for the rolling SLO view
            # and the accuracy events for the trust ledger. Both are
            # best-effort: a sink failure must not break the fetch.
            self._sink_health(health)
            self._sink_accuracy(accuracy_events)
            # Feed the sector-medians cohort so the next analyzer call
            # sees live multiples for any sector that's reached the
            # cohort floor (B-13). Best-effort — a refresh failure must
            # not break the fetch.
            self._sink_cohort(normalized)

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
        self._maybe_enqueue_for_retry(
            sym, health, chain_errors, retry_after_seconds=max_retry_after_seconds
        )
        # Raise an aggregate error so callers can decide whether to
        # 503, cache-fallback, or render a "no data" badge in the UI.
        # The retry-after hint travels on the exception so a downstream
        # worker scheduling a reschedule can honor the server cooldown
        # rather than fall back to default exponential backoff.
        raise ProviderChainExhausted(
            sym,
            tuple(health),
            tuple(chain_errors),
            retry_after_seconds=max_retry_after_seconds or None,
        )

    def _maybe_enqueue_for_retry(
        self,
        symbol: str,
        health: list[ProviderHealth],
        chain_errors: list[str],
        *,
        retry_after_seconds: float = 0.0,
    ) -> None:
        """Enqueue ``symbol`` when the chain's observed failures are
        all transient or rate-limited.

        Skips:

        * No retry queue wired (operator opted out via settings).
        * Any chain failure was non-transient — wouldn't recover on retry.
        * No actual failure rows (defensive; should never fire on the
          chain-exhausted path).

        ``retry_after_seconds`` is the maximum ``Retry-After`` value any
        rate-limited provider returned. Passed through so the queue can
        delay the first attempt accordingly instead of letting the
        worker hit the same upstream before its cooldown ends.
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
            self._retry_queue.enqueue(
                symbol,
                tuple(chain_errors),
                retry_after_seconds=retry_after_seconds or None,
            )
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
    """Every provider in the chain failed or was unconfigured.

    ``retry_after_seconds`` is the maximum ``Retry-After`` hint any
    rate-limited provider returned during this fetch (parsed by
    :func:`src.intelligence.fundamentals.http._parse_retry_after`).
    The retry worker forwards this to ``RetryQueue.record_failure`` so
    a server-supplied cooldown isn't undercut by our default backoff
    on the second attempt — RFC 6585 honored end-to-end, not just on
    the first enqueue.
    """

    def __init__(
        self,
        symbol: str,
        health: tuple[ProviderHealth, ...],
        errors: tuple[str, ...],
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        joined = "; ".join(errors) if errors else "no providers configured"
        super().__init__(
            f"fundamentals chain exhausted for {symbol}: {joined}"
        )
        self.symbol = symbol
        self.health = health
        self.errors = errors
        self.retry_after_seconds = retry_after_seconds


def _rejections_to_accuracy_events(
    provider: ProviderName,
    symbol: str,
    rejections: list[tuple[str, float]],
    observed_at: datetime,
) -> tuple[AccuracyEvent, ...]:
    """Turn :func:`sanitize_normalized` rejections into ledger events.

    Each rejection becomes one synthetic :class:`AccuracyEvent` with the
    ``provider`` as its own reference, ``agreed=False``, ``rel_error=1.0``,
    and a ``sanity:<field>`` namespace on the ``field`` column so the
    rows are filterable from cross-provider reconciliation events while
    still contributing to ``overall_accuracy`` aggregations.

    Empty input → empty tuple; the orchestrator's accuracy sink no-ops
    on empty input.
    """
    if not rejections:
        return ()
    return tuple(
        AccuracyEvent(
            provider=provider,
            reference_provider=provider,
            symbol=symbol,
            field=f"sanity:{field}",
            observed_value=value,
            reference_value=None,
            rel_error=1.0,
            agreed=False,
            fiscal_date=None,
            observed_at=observed_at,
        )
        for field, value in rejections
    )


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
