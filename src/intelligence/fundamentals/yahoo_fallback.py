"""Yahoo Finance fallback provider — last resort.

Scraped, unofficial, and openly brittle. Sits at the end of the chain
so it's only consulted when every keyed/structured provider has
failed. The mapper deliberately treats parse failures as
:class:`ProviderNotFound` rather than crashing — Yahoo is allowed to
silently disappear from the chain when their markup changes.

We use the v10 quoteSummary endpoint which returns structured JSON
(no HTML parsing). It's still scraping in the legal sense (no
official agreement), but the implementation is far more stable than
HTML scrape paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from src.config import Settings, get_settings
from src.intelligence.fundamentals.base import (
    ProviderNotFound,
    ProviderTransient,
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

_BASE_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary"

_MODULES = (
    "assetProfile",
    "summaryDetail",
    "defaultKeyStatistics",
    "financialData",
    "price",
)


class YahooFallbackProvider:
    """Last-resort Yahoo quoteSummary adapter."""

    name = ProviderName.YAHOO

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._base_url = base_url

    @property
    def is_configured(self) -> bool:
        # No credentials needed; always "configured" — but flagged in
        # warnings so the analyzer knows it's looking at scraped data.
        return True

    async def fetch(
        self, symbol: str
    ) -> tuple[NormalizedFundamentals, ProviderRawResponse]:
        sym = symbol.upper()
        headers = {
            # Yahoo blocks the default httpx UA; pretend to be a browser.
            "User-Agent": (
                "Mozilla/5.0 (compatible; EstherAnalyzer/0.1; "
                "+https://example.com)"
            ),
        }
        params = {"modules": ",".join(_MODULES)}

        data = await get_json(
            self.name,
            f"{self._base_url}/{sym}",
            params=params,
            headers=headers,
            client=self._client,
        )
        result = _first_result(data)
        if result is None:
            raise ProviderNotFound(self.name, f"no quoteSummary result for {sym}")

        try:
            profile = _map_profile(sym, result)
            ratios = _map_ratios(result)
            targets = _map_targets(result)
        except (KeyError, ValueError, TypeError) as exc:
            # Yahoo's shape rotates regularly — treat parsing failure as
            # transient rather than letting the orchestrator surface a
            # 500. The chain advances cleanly.
            raise ProviderTransient(self.name, f"yahoo parse error: {exc}") from exc

        fetched_at = datetime.now(UTC)
        normalized = NormalizedFundamentals(
            symbol=sym,
            fetched_at=fetched_at,
            primary_provider=self.name,
            contributing_providers=(self.name,),
            profile=profile,
            key_ratios=ratios,
            analyst_targets=targets,
            warnings=("Data sourced from Yahoo scrape — treat as best-effort.",),
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=sym,
            endpoint="quoteSummary",
            fetched_at=fetched_at,
            payload=data if isinstance(data, dict) else {"_raw": data},
        )
        return normalized, raw


def _first_result(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    qs = data.get("quoteSummary")
    if not isinstance(qs, dict):
        return None
    results = qs.get("result")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0]
    return None


def _raw_value(node: Any) -> Any:
    """Yahoo wraps numerics as ``{"raw": 1.23, "fmt": "1.23"}``."""
    if isinstance(node, dict):
        return node.get("raw")
    return node


def _map_profile(symbol: str, result: dict[str, Any]) -> CompanyProfile:
    asset = result.get("assetProfile") or {}
    price = result.get("price") or {}
    summary = result.get("summaryDetail") or {}
    key_stats = result.get("defaultKeyStatistics") or {}
    return CompanyProfile(
        symbol=symbol,
        name=price.get("longName") or price.get("shortName"),
        sector=asset.get("sector"),
        industry=asset.get("industry"),
        exchange=price.get("exchangeName"),
        country=asset.get("country"),
        currency=price.get("currency") or "USD",
        market_cap=_safe_float(_raw_value(summary.get("marketCap"))),
        enterprise_value=_safe_float(_raw_value(key_stats.get("enterpriseValue"))),
        shares_outstanding=_safe_float(_raw_value(key_stats.get("sharesOutstanding"))),
        description=asset.get("longBusinessSummary"),
    )


def _map_ratios(result: dict[str, Any]) -> KeyRatios:
    summary = result.get("summaryDetail") or {}
    key_stats = result.get("defaultKeyStatistics") or {}
    fin = result.get("financialData") or {}
    return KeyRatios(
        pe_ratio=_safe_float(_raw_value(summary.get("trailingPE"))),
        forward_pe=_safe_float(_raw_value(summary.get("forwardPE"))),
        peg_ratio=_safe_float(_raw_value(key_stats.get("pegRatio"))),
        price_to_sales=_safe_float(_raw_value(summary.get("priceToSalesTrailing12Months"))),
        price_to_book=_safe_float(_raw_value(key_stats.get("priceToBook"))),
        ev_to_ebitda=_safe_float(_raw_value(key_stats.get("enterpriseToEbitda"))),
        ev_to_revenue=_safe_float(_raw_value(key_stats.get("enterpriseToRevenue"))),
        debt_to_equity=_safe_float(_raw_value(fin.get("debtToEquity"))),
        current_ratio=_safe_float(_raw_value(fin.get("currentRatio"))),
        quick_ratio=_safe_float(_raw_value(fin.get("quickRatio"))),
        return_on_equity=_safe_float(_raw_value(fin.get("returnOnEquity"))),
        return_on_assets=_safe_float(_raw_value(fin.get("returnOnAssets"))),
        gross_margin=_safe_float(_raw_value(fin.get("grossMargins"))),
        operating_margin=_safe_float(_raw_value(fin.get("operatingMargins"))),
        net_margin=_safe_float(_raw_value(fin.get("profitMargins"))),
        dividend_yield=_safe_float(_raw_value(summary.get("dividendYield"))),
        payout_ratio=_safe_float(_raw_value(summary.get("payoutRatio"))),
        beta=_safe_float(_raw_value(summary.get("beta"))),
    )


def _map_targets(result: dict[str, Any]) -> AnalystTargets | None:
    fin = result.get("financialData") or {}
    high = _safe_float(_raw_value(fin.get("targetHighPrice")))
    low = _safe_float(_raw_value(fin.get("targetLowPrice")))
    mean = _safe_float(_raw_value(fin.get("targetMeanPrice")))
    median = _safe_float(_raw_value(fin.get("targetMedianPrice")))
    analysts = _safe_int(_raw_value(fin.get("numberOfAnalystOpinions")))
    rec = _safe_float(_raw_value(fin.get("recommendationMean")))
    if all(v is None for v in (high, low, mean, median)):
        return None
    return AnalystTargets(
        target_high=high,
        target_low=low,
        target_mean=mean,
        target_median=median,
        number_of_analysts=analysts,
        recommendation_mean=rec,
    )


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


__all__ = ["YahooFallbackProvider"]
