"""End-to-end tests for the provider accuracy ledger integration.

Exercises the full path that ships in this commit:

1. ``FundamentalsService.fetch()`` falls through to a secondary provider.
2. ``reconcile()`` emits :class:`AccuracyEvent`\\ s for each comparable field.
3. The service sinks those events into the :class:`AccuracyStore`.
4. The next ``fetch()`` looks up :class:`ProviderTrust` and multiplies
   the envelope ``provider_confidence`` by the resulting weight.

These tests use fake providers (no HTTP) and real on-disk stores so
the SQLite roundtrip is exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.data.accuracy_store import AccuracyStore
from src.data.health_store import HealthStore
from src.intelligence.fundamentals.base import ProviderUnavailable
from src.intelligence.fundamentals.models import (
    AccuracyEvent,
    BalanceSheet,
    CompanyProfile,
    IncomeStatement,
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)
from src.intelligence.fundamentals.provider_trust import ProviderTrust
from src.intelligence.fundamentals.service import FundamentalsService


class _FakeProvider:
    """Fake that returns either a configured NormalizedFundamentals or raises."""

    def __init__(
        self,
        name: ProviderName,
        *,
        configured: bool = True,
        income_revenue: float | None = None,
        income_net_income: float | None = None,
        income_eps: float | None = None,
        balance_debt: float | None = None,
        unavailable: bool = False,
    ) -> None:
        self.name = name
        self._configured = configured
        self._income_revenue = income_revenue
        self._income_net_income = income_net_income
        self._income_eps = income_eps
        self._balance_debt = balance_debt
        self._unavailable = unavailable
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def fetch(self, symbol: str) -> tuple[NormalizedFundamentals, Any]:
        self.calls += 1
        if self._unavailable:
            raise ProviderUnavailable(self.name, "configured failure")
        now = datetime.now(UTC)
        income = IncomeStatement(
            period=ReportPeriod.ANNUAL,
            fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
            revenue=self._income_revenue,
            net_income=self._income_net_income,
            eps_diluted=self._income_eps,
            eps_basic=None,
        )
        balance = BalanceSheet(
            period=ReportPeriod.ANNUAL,
            fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
            total_debt=self._balance_debt,
        )
        normalized = NormalizedFundamentals(
            symbol=symbol,
            fetched_at=now,
            primary_provider=self.name,
            contributing_providers=(self.name,),
            profile=CompanyProfile(symbol=symbol),
            income_statements=(income,),
            balance_sheets=(balance,),
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=symbol,
            endpoint="fake",
            fetched_at=now,
            payload={"ok": True},
        )
        return normalized, raw


@pytest.mark.asyncio
async def test_reconciliation_events_persist_to_accuracy_store(
    tmp_path: Path,
) -> None:
    """After a fetch that triggers reconciliation, the ledger has rows."""
    acc = AccuracyStore(tmp_path / "acc.db")
    health = HealthStore(tmp_path / "health.db")
    fmp = _FakeProvider(ProviderName.FMP, unavailable=True)
    # Finnhub becomes the primary (position 1) → triggers reconciliation
    # against the next configured provider (SEC EDGAR preferred).
    finnhub = _FakeProvider(
        ProviderName.FINNHUB,
        income_revenue=100.0,
        income_net_income=20.0,
        income_eps=2.0,
        balance_debt=50.0,
    )
    edgar = _FakeProvider(
        ProviderName.SEC_EDGAR,
        income_revenue=100.0,
        income_net_income=20.0,
        income_eps=2.0,
        balance_debt=50.0,
    )
    svc = FundamentalsService(
        providers=[fmp, finnhub, edgar],
        health_store=health,
        accuracy_store=acc,
    )

    result = await svc.fetch("AAPL")
    assert result.fundamentals.primary_provider is ProviderName.FINNHUB

    # 4 high-trust fields all agreed → 4 events persisted.
    summary = acc.summarize(ProviderName.FINNHUB, timedelta(days=30))
    assert summary.total_events == 4
    assert summary.total_agreed == 4
    assert summary.overall_accuracy == 1.0

    acc.close()
    health.close()


@pytest.mark.asyncio
async def test_sec_edgar_preferred_as_reconciliation_reference(
    tmp_path: Path,
) -> None:
    """When SEC EDGAR is configured, it's the secondary regardless of chain order.

    Even though Alpha Vantage is the next provider in the chain after
    Finnhub, EDGAR is preferred for the reconciliation reference
    because it's the official-filing source.
    """
    acc = AccuracyStore(tmp_path / "acc.db")
    fmp = _FakeProvider(ProviderName.FMP, unavailable=True)
    finnhub = _FakeProvider(
        ProviderName.FINNHUB,
        income_revenue=100.0,
        income_net_income=20.0,
        income_eps=2.0,
        balance_debt=50.0,
    )
    av = _FakeProvider(
        ProviderName.ALPHA_VANTAGE,
        income_revenue=200.0,
        income_net_income=40.0,
        income_eps=4.0,
        balance_debt=100.0,
    )
    edgar = _FakeProvider(
        ProviderName.SEC_EDGAR,
        income_revenue=100.0,
        income_net_income=20.0,
        income_eps=2.0,
        balance_debt=50.0,
    )
    svc = FundamentalsService(
        providers=[fmp, finnhub, av, edgar],
        accuracy_store=acc,
    )

    await svc.fetch("AAPL")
    # EDGAR was the reference, so events tag it as reference_provider.
    # Alpha Vantage should have ZERO calls — we skipped over it.
    assert edgar.calls == 1
    assert av.calls == 0
    summary = acc.summarize(ProviderName.FINNHUB, timedelta(days=30))
    assert summary.total_events == 4
    assert summary.total_agreed == 4  # because we compared against EDGAR
    acc.close()


@pytest.mark.asyncio
async def test_provider_trust_weight_multiplies_envelope_confidence(
    tmp_path: Path,
) -> None:
    """When a provider has 0% accuracy history, confidence drops below the
    chain-position baseline.

    Construct a service where Finnhub (position 1) has 50 prior
    disagreement events on file. The trust weight collapses to the
    floor (0.5); envelope confidence on the new fetch should reflect
    base_confidence (0.85 for position 1) × 0.5 ≈ 0.425, capped.
    """
    acc = AccuracyStore(tmp_path / "acc.db")
    # Seed the ledger with 50 disagreements for FINNHUB.
    now = datetime.now(UTC)
    acc.record(
        [
            AccuracyEvent(
                provider=ProviderName.FINNHUB,
                reference_provider=ProviderName.SEC_EDGAR,
                symbol="AAPL",
                field="revenue",
                observed_value=100.0,
                reference_value=200.0,
                rel_error=0.5,
                agreed=False,
                fiscal_date=datetime(2024, 12, 31, tzinfo=UTC),
                observed_at=now - timedelta(days=i),
            )
            for i in range(50)
        ]
    )

    # Chain: FMP fails, Finnhub wins. No EDGAR configured so no extra
    # reconciliation events get added on this fetch.
    fmp = _FakeProvider(ProviderName.FMP, unavailable=True)
    finnhub = _FakeProvider(
        ProviderName.FINNHUB,
        income_revenue=100.0,
        income_net_income=20.0,
        income_eps=2.0,
        balance_debt=50.0,
    )
    trust = ProviderTrust(
        accuracy_store=acc,
        health_store=None,
        ttl_seconds=300.0,
    )
    svc = FundamentalsService(
        providers=[fmp, finnhub],
        accuracy_store=acc,
        provider_trust=trust,
    )

    # Sanity check: trust weight should be at the floor.
    assert trust.weight_for(ProviderName.FINNHUB) == pytest.approx(0.5)

    result = await svc.fetch("AAPL")
    # base_confidence = 0.85 (position 1), no divergences (no secondary
    # configured), trust_weight = 0.5 → final = 0.425.
    assert result.envelope.provider_confidence == pytest.approx(0.425)
    acc.close()


@pytest.mark.asyncio
async def test_cold_start_provider_keeps_baseline_confidence(
    tmp_path: Path,
) -> None:
    """No prior history → trust weight = 1.0 → confidence equals baseline."""
    acc = AccuracyStore(tmp_path / "acc.db")
    fmp = _FakeProvider(
        ProviderName.FMP,
        income_revenue=100.0,
        income_net_income=20.0,
        income_eps=2.0,
        balance_debt=50.0,
    )
    svc = FundamentalsService(providers=[fmp], accuracy_store=acc)

    result = await svc.fetch("AAPL")
    # Position 0 (primary) → no reconciliation runs at all → confidence
    # is exactly the base 1.0 multiplied by the cold-start weight 1.0.
    assert result.envelope.provider_confidence == pytest.approx(1.0)
    acc.close()
