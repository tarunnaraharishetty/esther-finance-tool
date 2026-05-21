"""Orchestrator fallback-chain tests.

Uses fake providers (no HTTP) so we can drive the orchestrator
through every branch of its state machine deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.intelligence.fundamentals.base import (
    ProviderNotFound,
    ProviderRateLimited,
    ProviderTransient,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.models import (
    CompanyProfile,
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
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
