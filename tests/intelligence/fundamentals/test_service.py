"""Orchestrator fallback-chain tests.

Uses fake providers (no HTTP) so we can drive the orchestrator
through every branch of its state machine deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.intelligence.fundamentals.base import (
    ProviderNotFound,
    ProviderRateLimited,
    ProviderTransient,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.models import (
    CompanyProfile,
    IncomeStatement,
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)
from src.intelligence.fundamentals.service import (
    FundamentalsService,
    ProviderChainExhausted,
)


class _FakeProvider:
    """Hand-rolled fake — easier to control than a Mock here.

    Each call pops the next behavior off ``actions``. Behaviors:
    ``"ok"`` returns a normalized record + raw response; anything
    inheriting from BaseException is raised.
    """

    def __init__(
        self,
        name: ProviderName,
        actions: list[object],
        *,
        configured: bool = True,
    ) -> None:
        self.name = name
        self._actions = actions
        self._configured = configured
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def fetch(self, symbol: str):  # type: ignore[no-untyped-def]
        self.calls += 1
        action = self._actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if action == "ok":
            now = datetime.now(UTC)
            normalized = NormalizedFundamentals(
                symbol=symbol,
                fetched_at=now,
                primary_provider=self.name,
                contributing_providers=(self.name,),
                profile=CompanyProfile(symbol=symbol, name=f"{self.name.value} co"),
            )
            raw = ProviderRawResponse(
                provider=self.name,
                symbol=symbol,
                endpoint="fake",
                fetched_at=now,
                payload={"ok": True},
            )
            return normalized, raw
        raise AssertionError(f"unknown action: {action!r}")


@pytest.mark.asyncio
async def test_first_healthy_provider_short_circuits_chain() -> None:
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")

    assert result.fundamentals.primary_provider is ProviderName.FMP
    assert fmp.calls == 1
    assert finnhub.calls == 0  # second provider never consulted

    statuses = {(h.provider, h.status) for h in result.health}
    assert (ProviderName.FMP, "ok") in statuses
    # Skipped providers in the chain should still be reflected.
    assert (ProviderName.FINNHUB, "skipped") in statuses


@pytest.mark.asyncio
async def test_falls_through_rate_limited_to_next_provider() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderRateLimited(ProviderName.FMP, "429")]
    )
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB

    statuses = {(h.provider, h.status) for h in result.health}
    assert (ProviderName.FMP, "rate_limited") in statuses
    assert (ProviderName.FINNHUB, "ok") in statuses


@pytest.mark.asyncio
async def test_not_found_advances_to_next_provider() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderNotFound(ProviderName.FMP, "no data")]
    )
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("XYZ")
    statuses = {(h.provider, h.status) for h in result.health}
    assert (ProviderName.FMP, "empty") in statuses
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB


@pytest.mark.asyncio
async def test_unconfigured_provider_is_skipped_silently() -> None:
    fmp = _FakeProvider(ProviderName.FMP, [], configured=False)
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")
    assert fmp.calls == 0  # never called when unconfigured
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB

    statuses = {(h.provider, h.status) for h in result.health}
    assert (ProviderName.FMP, "unavailable") in statuses


@pytest.mark.asyncio
async def test_transient_failure_advances_chain() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderTransient(ProviderName.FMP, "boom")]
    )
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB
    statuses = {(h.provider, h.status) for h in result.health}
    assert (ProviderName.FMP, "transient") in statuses


@pytest.mark.asyncio
async def test_chain_exhausted_raises_with_health_log() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub = _FakeProvider(
        ProviderName.FINNHUB, [ProviderRateLimited(ProviderName.FINNHUB, "429")]
    )
    svc = FundamentalsService(providers=[fmp, finnhub])

    with pytest.raises(ProviderChainExhausted) as exc_info:
        await svc.fetch("AAPL")

    err = exc_info.value
    assert err.symbol == "AAPL"
    # Health log records both failures so the caller can persist them.
    assert {h.status for h in err.health} == {"unavailable", "rate_limited"}
    assert "fmp" in str(err)
    assert "finnhub" in str(err)


@pytest.mark.asyncio
async def test_contributing_providers_records_attempts() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderNotFound(ProviderName.FMP, "miss")]
    )
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")
    # contributing_providers should reflect the actual chain shape:
    # FMP was tried (and missed), Finnhub was the one that delivered.
    assert ProviderName.FMP in result.fundamentals.contributing_providers
    assert ProviderName.FINNHUB in result.fundamentals.contributing_providers


# -----------------------------------------------------------------------------
# Freshness envelope wiring
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_carries_source_chain_in_order_attempted() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderNotFound(ProviderName.FMP, "miss")]
    )
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")
    # source_chain is the providers actually attempted (skipped tail
    # is excluded), in the order they were tried.
    assert result.envelope.source_chain == ("fmp", "finnhub")


@pytest.mark.asyncio
async def test_envelope_provider_confidence_primary_is_one() -> None:
    """Position-0 success → confidence 1.0."""
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub])

    result = await svc.fetch("AAPL")
    assert result.envelope.provider_confidence == pytest.approx(1.00)


@pytest.mark.asyncio
async def test_envelope_provider_confidence_falls_with_chain_position() -> None:
    """Position-2 success → confidence 0.70 per the configured curve."""
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub = _FakeProvider(
        ProviderName.FINNHUB,
        [ProviderRateLimited(ProviderName.FINNHUB, "429")],
    )
    av = _FakeProvider(ProviderName.ALPHA_VANTAGE, ["ok"])
    svc = FundamentalsService(providers=[fmp, finnhub, av])

    result = await svc.fetch("AAPL")
    assert result.envelope.data.primary_provider is ProviderName.ALPHA_VANTAGE
    assert result.envelope.provider_confidence == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_envelope_as_of_falls_back_when_no_statements() -> None:
    """No statements → as_of = fetched_at minus 1 day, policy quarterly.

    The fake provider above ships profile-only records so this is the
    default path. Freshness should be ``fresh`` (1 day << 95 day window).
    """
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    svc = FundamentalsService(providers=[fmp])

    result = await svc.fetch("AAPL")
    age = result.envelope.fetched_at - result.envelope.as_of
    assert age == timedelta(days=1)
    assert result.envelope.freshness == "fresh"


@pytest.mark.asyncio
async def test_envelope_as_of_uses_latest_quarterly_fiscal_date() -> None:
    """A quarterly statement 60 days old should drive as_of and stay fresh."""
    now = datetime(2026, 5, 21, tzinfo=UTC)
    fiscal_date = now - timedelta(days=60)

    class _StatementProvider:
        name = ProviderName.FMP
        is_configured = True

        async def fetch(self, symbol: str):  # type: ignore[no-untyped-def]
            normalized = NormalizedFundamentals(
                symbol=symbol,
                fetched_at=now,
                primary_provider=self.name,
                contributing_providers=(self.name,),
                profile=CompanyProfile(symbol=symbol, name="test"),
                income_statements=(
                    IncomeStatement(
                        period=ReportPeriod.QUARTERLY,
                        fiscal_date=fiscal_date,
                        revenue=1_000.0,
                    ),
                ),
            )
            raw = ProviderRawResponse(
                provider=self.name,
                symbol=symbol,
                endpoint="fake",
                fetched_at=now,
                payload={},
            )
            return normalized, raw

    svc = FundamentalsService(providers=[_StatementProvider()])
    result = await svc.fetch("AAPL")

    # as_of comes from the quarterly fiscal_date; freshness is fresh
    # (60 days < 95-day quarterly fresh window).
    assert result.envelope.as_of.date() == fiscal_date.date()
    assert result.envelope.freshness == "fresh"


@pytest.mark.asyncio
async def test_envelope_freshness_expired_for_year_old_statement() -> None:
    """A quarterly statement >400 days old should grade as expired."""
    very_old = datetime(2026, 5, 21, tzinfo=UTC) - timedelta(days=420)

    class _StaleStatementProvider:
        name = ProviderName.FMP
        is_configured = True

        async def fetch(self, symbol: str):  # type: ignore[no-untyped-def]
            now = datetime.now(UTC)
            normalized = NormalizedFundamentals(
                symbol=symbol,
                fetched_at=now,
                primary_provider=self.name,
                contributing_providers=(self.name,),
                profile=CompanyProfile(symbol=symbol, name="test"),
                income_statements=(
                    IncomeStatement(
                        period=ReportPeriod.QUARTERLY,
                        fiscal_date=very_old,
                        revenue=1.0,
                    ),
                ),
            )
            raw = ProviderRawResponse(
                provider=self.name,
                symbol=symbol,
                endpoint="fake",
                fetched_at=now,
                payload={},
            )
            return normalized, raw

    svc = FundamentalsService(providers=[_StaleStatementProvider()])
    result = await svc.fetch("AAPL")
    assert result.envelope.freshness == "expired"


@pytest.mark.asyncio
async def test_fundamentals_property_unwraps_envelope_data() -> None:
    """`.fundamentals` is the convenience accessor for envelope.data."""
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    svc = FundamentalsService(providers=[fmp])
    result = await svc.fetch("AAPL")
    assert result.fundamentals is result.envelope.data


# -----------------------------------------------------------------------------
# Health-store sink
# -----------------------------------------------------------------------------


class _CapturingHealthStore:
    """In-memory health-store stub for sink tests.

    Records every batch passed to ``record()`` so we can assert on
    the wire shape without spinning up SQLite.
    """

    def __init__(self, *, raise_on_record: bool = False) -> None:
        self.batches: list[tuple[object, ...]] = []
        self._raise = raise_on_record

    def record(self, rows: object) -> int:
        if self._raise:
            raise RuntimeError("simulated DB failure")
        rows_tuple = tuple(rows)  # type: ignore[arg-type]
        self.batches.append(rows_tuple)
        return len(rows_tuple)


@pytest.mark.asyncio
async def test_successful_fetch_writes_health_rows_to_store() -> None:
    """Per-call health log must land in the store on the success path."""
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    finnhub = _FakeProvider(ProviderName.FINNHUB, ["ok"])
    store = _CapturingHealthStore()
    svc = FundamentalsService(providers=[fmp, finnhub], health_store=store)

    await svc.fetch("AAPL")

    assert len(store.batches) == 1
    batch = store.batches[0]
    statuses = {row.status for row in batch}
    # FMP delivered → ok; Finnhub got skipped because the chain stopped.
    assert "ok" in statuses
    assert "skipped" in statuses


@pytest.mark.asyncio
async def test_chain_exhausted_still_writes_health_rows() -> None:
    """Failure rows are the most useful operator signal — never drop them."""
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub = _FakeProvider(
        ProviderName.FINNHUB,
        [ProviderRateLimited(ProviderName.FINNHUB, "429")],
    )
    store = _CapturingHealthStore()
    svc = FundamentalsService(providers=[fmp, finnhub], health_store=store)

    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("XYZ")

    assert len(store.batches) == 1
    statuses = {row.status for row in store.batches[0]}
    assert statuses == {"unavailable", "rate_limited"}


@pytest.mark.asyncio
async def test_health_store_write_failure_does_not_break_fetch() -> None:
    """A broken sink must NEVER take down the fetch path.

    Observability is a nice-to-have; a fetch that erupts in the
    middle of a research session is a feature outage. Order of
    priorities is explicit in the service.
    """
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    store = _CapturingHealthStore(raise_on_record=True)
    svc = FundamentalsService(providers=[fmp], health_store=store)

    # If the sink had been allowed to propagate, this would raise.
    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FMP


@pytest.mark.asyncio
async def test_no_health_store_means_no_sink_attempted() -> None:
    """Service must handle ``health_store=None`` cleanly (default path)."""
    fmp = _FakeProvider(ProviderName.FMP, ["ok"])
    svc = FundamentalsService(providers=[fmp], health_store=None)
    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FMP


# -----------------------------------------------------------------------------
# Reconciliation
# -----------------------------------------------------------------------------


class _ConfiguredProvider:
    """Provider stub returning fully configured NormalizedFundamentals.

    Used by reconciliation tests where the comparison needs real
    revenue / net_income / EPS / debt values, not the profile-only
    payloads :class:`_FakeProvider` ships.
    """

    def __init__(
        self,
        name: ProviderName,
        *,
        revenue: float = 100.0,
        net_income: float = 20.0,
        eps_diluted: float = 2.0,
        total_debt: float = 50.0,
        raise_with: BaseException | None = None,
    ) -> None:
        self.name = name
        self._raise = raise_with
        self._revenue = revenue
        self._net_income = net_income
        self._eps_diluted = eps_diluted
        self._total_debt = total_debt
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return True

    async def fetch(self, symbol: str):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        from src.intelligence.fundamentals.models import (
            BalanceSheet,
            CompanyProfile,
            IncomeStatement,
        )

        now = datetime.now(UTC)
        normalized = NormalizedFundamentals(
            symbol=symbol,
            fetched_at=now,
            primary_provider=self.name,
            contributing_providers=(self.name,),
            profile=CompanyProfile(symbol=symbol, name=f"{self.name.value} co"),
            income_statements=(
                IncomeStatement(
                    period=ReportPeriod.ANNUAL,
                    fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
                    revenue=self._revenue,
                    net_income=self._net_income,
                    eps_diluted=self._eps_diluted,
                ),
            ),
            balance_sheets=(
                BalanceSheet(
                    period=ReportPeriod.ANNUAL,
                    fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
                    total_debt=self._total_debt,
                ),
            ),
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=symbol,
            endpoint="fake",
            fetched_at=now,
            payload={},
        )
        return normalized, raw


@pytest.mark.asyncio
async def test_reconciliation_fires_on_fallback_chain_position() -> None:
    """Position-1 primary triggers a secondary call against position 2."""
    fmp_fail = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub_primary = _ConfiguredProvider(ProviderName.FINNHUB, revenue=100.0)
    av_secondary = _ConfiguredProvider(ProviderName.ALPHA_VANTAGE, revenue=120.0)
    svc = FundamentalsService(
        providers=[fmp_fail, finnhub_primary, av_secondary]
    )
    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB
    # Reconciliation ran against alpha_vantage and found revenue divergence.
    assert av_secondary.calls == 1
    fields = {d.field for d in result.divergences}
    assert "revenue" in fields


@pytest.mark.asyncio
async def test_reconciliation_skipped_when_primary_is_position_zero() -> None:
    """Happy path stays single-call — no double API spend."""
    fmp_primary = _ConfiguredProvider(ProviderName.FMP, revenue=100.0)
    finnhub_secondary = _ConfiguredProvider(ProviderName.FINNHUB, revenue=999.0)
    svc = FundamentalsService(providers=[fmp_primary, finnhub_secondary])
    result = await svc.fetch("AAPL")
    assert finnhub_secondary.calls == 0
    assert result.divergences == ()


@pytest.mark.asyncio
async def test_reconciliation_skipped_when_disabled_in_settings() -> None:
    fmp_fail = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub_primary = _ConfiguredProvider(ProviderName.FINNHUB, revenue=100.0)
    av_secondary = _ConfiguredProvider(ProviderName.ALPHA_VANTAGE, revenue=999.0)
    svc = FundamentalsService(
        providers=[fmp_fail, finnhub_primary, av_secondary]
    )
    svc._settings.reconciliation_enabled = False  # type: ignore[misc]
    result = await svc.fetch("AAPL")
    assert av_secondary.calls == 0
    assert result.divergences == ()


@pytest.mark.asyncio
async def test_reconciliation_skipped_when_no_secondary_configured() -> None:
    """No provider after primary is configured → no reconciliation call."""
    fmp_fail = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub_primary = _ConfiguredProvider(ProviderName.FINNHUB, revenue=100.0)
    av_unconfigured = _FakeProvider(
        ProviderName.ALPHA_VANTAGE, [], configured=False
    )
    svc = FundamentalsService(
        providers=[fmp_fail, finnhub_primary, av_unconfigured]
    )
    result = await svc.fetch("AAPL")
    assert result.divergences == ()


@pytest.mark.asyncio
async def test_reconciliation_secondary_failure_does_not_break_fetch() -> None:
    """Secondary throws → primary result still returned, divergences empty."""
    fmp_fail = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub_primary = _ConfiguredProvider(ProviderName.FINNHUB, revenue=100.0)
    av_broken = _ConfiguredProvider(
        ProviderName.ALPHA_VANTAGE, raise_with=RuntimeError("simulated")
    )
    svc = FundamentalsService(
        providers=[fmp_fail, finnhub_primary, av_broken]
    )
    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB
    assert result.divergences == ()


@pytest.mark.asyncio
async def test_divergences_penalize_provider_confidence() -> None:
    """Confidence drops by 0.85 per divergence, floored at 0.5x base."""
    fmp_fail = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    # 4 divergent fields → penalty 0.85**4 = 0.522, above the 0.5 floor.
    finnhub_primary = _ConfiguredProvider(
        ProviderName.FINNHUB,
        revenue=100.0,
        net_income=10.0,
        eps_diluted=1.0,
        total_debt=50.0,
    )
    av_secondary = _ConfiguredProvider(
        ProviderName.ALPHA_VANTAGE,
        revenue=200.0,
        net_income=20.0,
        eps_diluted=2.0,
        total_debt=200.0,
    )
    svc = FundamentalsService(
        providers=[fmp_fail, finnhub_primary, av_secondary]
    )
    result = await svc.fetch("AAPL")
    # Base confidence at position 1 is 0.85; with 4 divergences,
    # final = 0.85 * 0.85**4 = 0.85 * 0.522 ≈ 0.444 → floored to 0.85 * 0.5 = 0.425.
    assert len(result.divergences) == 4
    assert result.envelope.provider_confidence < 0.85
    # Floor sanity: not below 0.5 * base = 0.425.
    assert result.envelope.provider_confidence >= 0.85 * 0.5 - 1e-9


# -----------------------------------------------------------------------------
# Retry-queue enqueue policy
# -----------------------------------------------------------------------------


class _CapturingRetryQueue:
    """Minimal stub matching ``RetryQueue.enqueue`` signature."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.last_retry_after: float | None = None

    def enqueue(
        self,
        symbol: str,
        errors: tuple[str, ...],
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        # Capture the Retry-After hint too so tests can assert that the
        # service propagates it from rate-limited providers into the
        # queue (BUGS.md B-10).
        self.calls.append((symbol.upper(), tuple(errors)))
        self.last_retry_after = retry_after_seconds


@pytest.mark.asyncio
async def test_enqueues_on_transient_only_chain_exhaustion() -> None:
    """All-transient failure → enqueue + raise."""
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderRateLimited(ProviderName.FMP, "429")]
    )
    finnhub = _FakeProvider(
        ProviderName.FINNHUB, [ProviderTransient(ProviderName.FINNHUB, "boom")]
    )
    queue = _CapturingRetryQueue()
    svc = FundamentalsService(providers=[fmp, finnhub], retry_queue=queue)
    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("AAPL")
    assert len(queue.calls) == 1
    assert queue.calls[0][0] == "AAPL"


@pytest.mark.asyncio
async def test_does_not_enqueue_when_any_error_is_permanent() -> None:
    """Permanent failure mixed in → no enqueue (won't recover on retry)."""
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    finnhub = _FakeProvider(
        ProviderName.FINNHUB, [ProviderRateLimited(ProviderName.FINNHUB, "429")]
    )
    queue = _CapturingRetryQueue()
    svc = FundamentalsService(providers=[fmp, finnhub], retry_queue=queue)
    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("AAPL")
    assert queue.calls == []


@pytest.mark.asyncio
async def test_does_not_enqueue_when_only_unconfigured_providers() -> None:
    """Empty-key case: providers all 'unavailable' (config gap) → no enqueue."""
    fmp = _FakeProvider(ProviderName.FMP, [], configured=False)
    finnhub = _FakeProvider(ProviderName.FINNHUB, [], configured=False)
    queue = _CapturingRetryQueue()
    svc = FundamentalsService(providers=[fmp, finnhub], retry_queue=queue)
    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("AAPL")
    assert queue.calls == []


@pytest.mark.asyncio
async def test_enqueue_failure_does_not_break_the_caller() -> None:
    """A broken queue must not mask the original ProviderChainExhausted."""

    class _BrokenQueue:
        def enqueue(self, *_args: object) -> None:
            raise RuntimeError("simulated queue failure")

    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderRateLimited(ProviderName.FMP, "429")]
    )
    svc = FundamentalsService(providers=[fmp], retry_queue=_BrokenQueue())
    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("AAPL")


@pytest.mark.asyncio
async def test_no_retry_queue_means_no_enqueue_attempted() -> None:
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderRateLimited(ProviderName.FMP, "429")]
    )
    svc = FundamentalsService(providers=[fmp], retry_queue=None)
    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("AAPL")


@pytest.mark.asyncio
async def test_chain_exhausted_carries_max_retry_after_hint() -> None:
    """The largest Retry-After across the chain rides the exception so the
    worker reschedule honors the server cooldown on the second attempt."""
    fmp = _FakeProvider(
        ProviderName.FMP,
        [ProviderRateLimited(ProviderName.FMP, "429", retry_after_seconds=120.0)],
    )
    finnhub = _FakeProvider(
        ProviderName.FINNHUB,
        [ProviderRateLimited(ProviderName.FINNHUB, "429", retry_after_seconds=600.0)],
    )
    svc = FundamentalsService(providers=[fmp, finnhub], retry_queue=None)
    with pytest.raises(ProviderChainExhausted) as exc_info:
        await svc.fetch("AAPL")
    assert exc_info.value.retry_after_seconds == 600.0


@pytest.mark.asyncio
async def test_chain_exhausted_retry_after_is_none_when_no_hint() -> None:
    """No provider supplied Retry-After → exception carries None, not 0.0."""
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderTransient(ProviderName.FMP, "boom")]
    )
    svc = FundamentalsService(providers=[fmp], retry_queue=None)
    with pytest.raises(ProviderChainExhausted) as exc_info:
        await svc.fetch("AAPL")
    assert exc_info.value.retry_after_seconds is None


# ---------------------------------------------------------------------------
# B-11: sanity rejections feed the accuracy ledger
# ---------------------------------------------------------------------------


class _CapturingAccuracyStore:
    """Stub that records every batch of AccuracyEvents sunk by the service."""

    def __init__(self) -> None:
        self.batches: list[tuple[object, ...]] = []

    def record(self, events: object) -> int:
        materialized = tuple(events)
        self.batches.append(materialized)
        return len(materialized)


class _BogusProvider:
    """Provider stub that ships an obviously bogus market_cap.

    Used to drive the sanity path without coupling to the full
    NormalizedFundamentals shape — only the one field under test is
    interesting here.
    """

    def __init__(self, name: ProviderName) -> None:
        self.name = name

    @property
    def is_configured(self) -> bool:
        return True

    async def fetch(self, symbol: str):  # type: ignore[no-untyped-def]
        from src.intelligence.fundamentals.models import CompanyProfile

        now = datetime.now(UTC)
        normalized = NormalizedFundamentals(
            symbol=symbol,
            fetched_at=now,
            primary_provider=self.name,
            contributing_providers=(self.name,),
            profile=CompanyProfile(symbol=symbol, market_cap=-42.0),
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=symbol,
            endpoint="bogus",
            fetched_at=now,
            payload={},
        )
        return normalized, raw


@pytest.mark.asyncio
async def test_sanity_rejection_emits_accuracy_event() -> None:
    """A provider that ships a negative market cap should log one
    self-referential AccuracyEvent with the ``sanity:`` namespace and
    agreed=False so the trust weight reflects the failure."""
    store = _CapturingAccuracyStore()
    svc = FundamentalsService(
        providers=[_BogusProvider(ProviderName.FMP)], accuracy_store=store
    )
    result = await svc.fetch("AAPL")
    # The bad scalar should have been dropped on the way out.
    assert result.fundamentals.profile.market_cap is None
    # Exactly one batch of accuracy events was sunk; it must contain at
    # least one sanity-namespaced event for this rejection.
    flat = [event for batch in store.batches for event in batch]
    sanity = [e for e in flat if e.field.startswith("sanity:")]
    assert sanity, "expected at least one sanity:* accuracy event"
    event = next(e for e in sanity if "market_cap" in e.field)
    assert event.provider is ProviderName.FMP
    assert event.reference_provider is ProviderName.FMP  # self sentinel
    assert event.agreed is False
    assert event.rel_error == 1.0
    assert event.observed_value == -42.0
    assert event.reference_value is None


@pytest.mark.asyncio
async def test_clean_provider_emits_no_sanity_events() -> None:
    """Provider with no bogus values: no sanity events on the ledger."""
    store = _CapturingAccuracyStore()
    svc = FundamentalsService(
        providers=[_ConfiguredProvider(ProviderName.FMP)],
        accuracy_store=store,
    )
    await svc.fetch("AAPL")
    flat = [event for batch in store.batches for event in batch]
    assert all(not e.field.startswith("sanity:") for e in flat)


# ---------------------------------------------------------------------------
# B-13: every successful fetch feeds the sector cohort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_fetch_observes_into_sector_cohort() -> None:
    """B-13 end-to-end: a successful fetch must reach the cohort so
    enough observations move ``lookup()`` off the static seed."""
    from src.intelligence.analyzer.sector_medians import SectorCohort

    cohort = SectorCohort()
    svc = FundamentalsService(
        providers=[_ConfiguredProvider(ProviderName.FMP)],
        sector_cohort=cohort,
    )
    await svc.fetch("AAPL")
    snapshot = cohort.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_failed_fetch_does_not_pollute_cohort() -> None:
    """A provider chain exhaustion produces no cohort observation —
    we only seed the medians from confirmed-healthy data."""
    from src.intelligence.analyzer.sector_medians import SectorCohort

    cohort = SectorCohort()
    fmp = _FakeProvider(
        ProviderName.FMP, [ProviderTransient(ProviderName.FMP, "boom")]
    )
    svc = FundamentalsService(providers=[fmp], sector_cohort=cohort)
    with pytest.raises(ProviderChainExhausted):
        await svc.fetch("AAPL")
    assert cohort.snapshot() == []


@pytest.mark.asyncio
async def test_cohort_observe_failure_does_not_break_fetch() -> None:
    """A misbehaving cohort hook must never break a real fetch — the
    medians override is a confidence modifier, not a correctness gate."""
    from src.intelligence.analyzer.sector_medians import SectorCohort

    class _BrokenCohort(SectorCohort):
        def observe(self, record):  # type: ignore[no-untyped-def, override]
            raise RuntimeError("simulated cohort failure")

    svc = FundamentalsService(
        providers=[_ConfiguredProvider(ProviderName.FMP)],
        sector_cohort=_BrokenCohort(),
    )
    # Must not raise.
    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FMP


@pytest.mark.asyncio
async def test_no_divergences_leaves_provider_confidence_intact() -> None:
    fmp_fail = _FakeProvider(
        ProviderName.FMP, [ProviderUnavailable(ProviderName.FMP, "no key")]
    )
    # Identical secondary data → no divergences → no penalty.
    finnhub_primary = _ConfiguredProvider(ProviderName.FINNHUB, revenue=100.0)
    av_secondary = _ConfiguredProvider(ProviderName.ALPHA_VANTAGE, revenue=100.0)
    svc = FundamentalsService(
        providers=[fmp_fail, finnhub_primary, av_secondary]
    )
    result = await svc.fetch("AAPL")
    assert result.divergences == ()
    # Position 1 base confidence is 0.85; no penalty applied.
    assert result.envelope.provider_confidence == pytest.approx(0.85)
