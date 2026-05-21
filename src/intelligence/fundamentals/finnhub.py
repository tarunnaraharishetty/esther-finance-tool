"""Finnhub provider.

Finnhub's free tier exposes:

* ``/stock/profile2``               — profile (name, exchange, market cap)
* ``/stock/metric?metric=all``      — TTM key ratios (and many more)
* ``/stock/financials-reported``    — full statements (US only, FY/Q)
* ``/stock/recommendation``         — analyst recommendation history
* ``/stock/price-target``           — target price consensus

We focus on profile + metrics + targets here. Full reported financials
exist on the API but the schema is XBRL-styled and easier to source
from SEC EDGAR. This provider intentionally leaves
``income_statements`` / ``balance_sheets`` / ``cash_flows`` empty —
the orchestrator can merge them in from a later provider if needed
(Phase 2 refinement).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

from src.config import Settings, get_settings
from src.intelligence.fundamentals.base import (
    ProviderNotFound,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.http import get_json
from src.intelligence.fundamentals.models import (
    AnalystTargets,
    CompanyProfile,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
)

_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubProvider:
    """Finnhub adapter."""

    name = ProviderName.FINNHUB

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._base_url = base_url.rstrip("/")

    @property
    def is_configured(self) -> bool:
        return self._settings.finnhub_api_key is not None

    async def fetch(
        self, symbol: str
    ) -> tuple[NormalizedFundamentals, ProviderRawResponse]:
        if not self.is_configured:
            raise ProviderUnavailable(self.name, "FINNHUB_API_KEY not set")
        sym = symbol.upper()
        api_key = self._settings.finnhub_api_key.get_secret_value()  # type: ignore[union-attr]

        async def _get(path: str, **params: Any) -> Any:
            return await get_json(
                self.name,
                f"{self._base_url}{path}",
                params={"token": api_key, "symbol": sym, **params},
                client=self._client,
            )

        profile_raw, metric_raw, target_raw = await asyncio.gather(
            _get("/stock/profile2"),
            _get("/stock/metric", metric="all"),
            _get("/stock/price-target"),
            return_exceptions=True,
        )

        if isinstance(profile_raw, ProviderNotFound):
            raise profile_raw
        if isinstance(profile_raw, BaseException):
            raise profile_raw

        profile = _map_profile(sym, profile_raw)
        ratios = _map_ratios(metric_raw)
        targets = _map_targets(target_raw)

        fetched_at = datetime.now(UTC)
        normalized = NormalizedFundamentals(
            symbol=sym,
            fetched_at=fetched_at,
            primary_provider=self.name,
            contributing_providers=(self.name,),
            profile=profile,
            key_ratios=ratios,
            analyst_targets=targets,
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=sym,
            endpoint="/stock/profile2,/stock/metric,/stock/price-target",
            fetched_at=fetched_at,
            payload={
                "profile": _coerce_payload(profile_raw),
                "metric": _coerce_payload(metric_raw),
                "price_target": _coerce_payload(target_raw),
            },
        )
        return normalized, raw


def _coerce_payload(value: Any) -> Any:
    if isinstance(value, BaseException):
        return {"_error": type(value).__name__, "message": str(value)}
    return value


def _map_profile(symbol: str, raw: Any) -> CompanyProfile:
    if not isinstance(raw, dict):
        return CompanyProfile(symbol=symbol)
    return CompanyProfile(
        symbol=symbol,
        name=raw.get("name"),
        sector=raw.get("finnhubIndustry"),
        industry=raw.get("finnhubIndustry"),
        exchange=raw.get("exchange"),
        country=raw.get("country"),
        currency=raw.get("currency") or "USD",
        market_cap=_safe_float(raw.get("marketCapitalization")),
        shares_outstanding=_safe_float(raw.get("shareOutstanding")),
        description=None,
    )


def _map_ratios(raw: Any) -> KeyRatios:
    if not isinstance(raw, dict):
        return KeyRatios()
    metric = raw.get("metric")
    if not isinstance(metric, dict):
        return KeyRatios()
    return KeyRatios(
        pe_ratio=_safe_float(metric.get("peNormalizedAnnual"))
        or _safe_float(metric.get("peTTM")),
        forward_pe=_safe_float(metric.get("peExclExtraTTM")),
        peg_ratio=_safe_float(metric.get("pegRatio")),
        price_to_sales=_safe_float(metric.get("psTTM")),
        price_to_book=_safe_float(metric.get("pbAnnual")),
        ev_to_ebitda=_safe_float(metric.get("currentEv/freeCashFlowTTM"))
        or _safe_float(metric.get("enterpriseValue/ebitdaTTM")),
        ev_to_revenue=_safe_float(metric.get("enterpriseValue/revenueTTM")),
        debt_to_equity=_safe_float(metric.get("totalDebt/totalEquityAnnual")),
        current_ratio=_safe_float(metric.get("currentRatioAnnual")),
        quick_ratio=_safe_float(metric.get("quickRatioAnnual")),
        return_on_equity=_safe_float(metric.get("roeTTM")),
        return_on_assets=_safe_float(metric.get("roaTTM")),
        gross_margin=_safe_float(metric.get("grossMarginTTM")),
        operating_margin=_safe_float(metric.get("operatingMarginTTM")),
        net_margin=_safe_float(metric.get("netProfitMarginTTM")),
        dividend_yield=_safe_float(metric.get("dividendYieldIndicatedAnnual")),
        payout_ratio=_safe_float(metric.get("payoutRatioTTM")),
        beta=_safe_float(metric.get("beta")),
    )


def _map_targets(raw: Any) -> AnalystTargets | None:
    if not isinstance(raw, dict) or not raw.get("lastUpdated"):
        return None
    targets = AnalystTargets(
        target_high=_safe_float(raw.get("targetHigh")),
        target_low=_safe_float(raw.get("targetLow")),
        target_mean=_safe_float(raw.get("targetMean")),
        target_median=_safe_float(raw.get("targetMedian")),
        number_of_analysts=_safe_int(raw.get("numberOfAnalysts")),
    )
    if all(
        v is None
        for v in (
            targets.target_high,
            targets.target_low,
            targets.target_mean,
            targets.target_median,
        )
    ):
        return None
    return targets


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["FinnhubProvider"]
