"""Financial Modeling Prep (FMP) provider.

FMP's free tier covers US equities and returns a clean, multi-period
shape per statement kind. We fetch four endpoints in parallel:

* ``/api/v3/profile/{symbol}``           — company profile + market cap
* ``/api/v3/income-statement/{symbol}``  — annual income statements
* ``/api/v3/balance-sheet-statement/...``— annual balance sheets
* ``/api/v3/cash-flow-statement/...``    — annual cash flows
* ``/api/v3/key-metrics-ttm/{symbol}``   — TTM ratios
* ``/api/v3/price-target-consensus``     — analyst targets (premium-only on
  some plans; treated as optional)

The mappers below tolerate partial responses — any missing field
maps to ``None`` rather than KeyError.
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
    BalanceSheet,
    CashFlow,
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)

_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FmpProvider:
    """Financial Modeling Prep adapter."""

    name = ProviderName.FMP

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = _BASE_URL,
        annual_limit: int = 5,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._annual_limit = annual_limit

    @property
    def is_configured(self) -> bool:
        return self._settings.fmp_api_key is not None

    async def fetch(
        self, symbol: str
    ) -> tuple[NormalizedFundamentals, ProviderRawResponse]:
        if not self.is_configured:
            raise ProviderUnavailable(self.name, "FMP_API_KEY not set")
        sym = symbol.upper()
        api_key = self._settings.fmp_api_key.get_secret_value()  # type: ignore[union-attr]

        async def _get(endpoint: str, **params: Any) -> Any:
            return await get_json(
                self.name,
                f"{self._base_url}{endpoint}",
                params={"apikey": api_key, **params},
                client=self._client,
            )

        profile_task = _get(f"/profile/{sym}")
        income_task = _get(f"/income-statement/{sym}", limit=self._annual_limit)
        balance_task = _get(
            f"/balance-sheet-statement/{sym}", limit=self._annual_limit
        )
        cashflow_task = _get(f"/cash-flow-statement/{sym}", limit=self._annual_limit)
        ratios_task = _get(f"/key-metrics-ttm/{sym}", limit=1)
        targets_task = self._fetch_targets(sym, api_key)

        (
            profile_raw,
            income_raw,
            balance_raw,
            cashflow_raw,
            ratios_raw,
            targets_raw,
        ) = await asyncio.gather(
            profile_task,
            income_task,
            balance_task,
            cashflow_task,
            ratios_task,
            targets_task,
            return_exceptions=True,
        )

        # The profile endpoint is the only one we treat as mandatory —
        # if FMP doesn't know the company at all, fall through to the
        # next provider. Statement endpoints are allowed to fail
        # individually (e.g. for very new IPOs that lack history).
        if isinstance(profile_raw, ProviderNotFound):
            raise profile_raw
        if isinstance(profile_raw, BaseException):
            raise profile_raw  # auth / transient / unavailable — orchestrator handles

        profile = _map_profile(sym, profile_raw)
        income = _map_income(income_raw)
        balance = _map_balance(balance_raw)
        cashflow = _map_cashflow(cashflow_raw)
        ratios = _map_ratios(ratios_raw)
        targets = _map_targets(targets_raw)

        fetched_at = datetime.now(UTC)
        normalized = NormalizedFundamentals(
            symbol=sym,
            fetched_at=fetched_at,
            primary_provider=self.name,
            contributing_providers=(self.name,),
            profile=profile,
            income_statements=income,
            balance_sheets=balance,
            cash_flows=cashflow,
            key_ratios=ratios,
            analyst_targets=targets,
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=sym,
            endpoint="/profile,/income-statement,/balance-sheet-statement,"
            "/cash-flow-statement,/key-metrics-ttm,/price-target-consensus",
            fetched_at=fetched_at,
            payload={
                "profile": _coerce_payload(profile_raw),
                "income_statements": _coerce_payload(income_raw),
                "balance_sheets": _coerce_payload(balance_raw),
                "cash_flows": _coerce_payload(cashflow_raw),
                "key_metrics_ttm": _coerce_payload(ratios_raw),
                "price_target_consensus": _coerce_payload(targets_raw),
            },
        )
        return normalized, raw

    async def _fetch_targets(self, sym: str, api_key: str) -> Any:
        """Best-effort analyst-target fetch.

        Many FMP plans gate this endpoint. Returning ``None`` rather
        than propagating an Unavailable lets the rest of the pipeline
        proceed without targets.
        """
        try:
            return await get_json(
                self.name,
                f"{self._base_url}/price-target-consensus",
                params={"symbol": sym, "apikey": api_key},
                client=self._client,
            )
        except (ProviderUnavailable, ProviderNotFound):
            return None


def _coerce_payload(value: Any) -> Any:
    """Replace exceptions with a small error marker so the raw blob is JSON-safe."""
    if isinstance(value, BaseException):
        return {"_error": type(value).__name__, "message": str(value)}
    return value


def _map_profile(symbol: str, raw: Any) -> CompanyProfile:
    record = _first_record(raw)
    if record is None:
        return CompanyProfile(symbol=symbol)
    return CompanyProfile(
        symbol=symbol,
        name=record.get("companyName"),
        sector=record.get("sector"),
        industry=record.get("industry"),
        exchange=record.get("exchangeShortName") or record.get("exchange"),
        country=record.get("country"),
        currency=record.get("currency") or "USD",
        market_cap=_safe_float(record.get("mktCap")),
        enterprise_value=None,
        shares_outstanding=_safe_float(record.get("sharesOutstanding")),
        description=record.get("description"),
        cik=record.get("cik"),
    )


def _map_income(raw: Any) -> tuple[IncomeStatement, ...]:
    if isinstance(raw, BaseException) or not isinstance(raw, list):
        return ()
    out: list[IncomeStatement] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        out.append(
            IncomeStatement(
                period=_period_from(entry.get("period")),
                fiscal_date=_parse_date(entry.get("date")),
                revenue=_safe_float(entry.get("revenue")),
                cost_of_revenue=_safe_float(entry.get("costOfRevenue")),
                gross_profit=_safe_float(entry.get("grossProfit")),
                operating_income=_safe_float(entry.get("operatingIncome")),
                ebitda=_safe_float(entry.get("ebitda")),
                ebit=_safe_float(entry.get("operatingIncome")),
                net_income=_safe_float(entry.get("netIncome")),
                eps_basic=_safe_float(entry.get("eps")),
                eps_diluted=_safe_float(entry.get("epsdiluted")),
                shares_diluted=_safe_float(entry.get("weightedAverageShsOutDil")),
                reported_currency=entry.get("reportedCurrency") or "USD",
            )
        )
    return tuple(out)


def _map_balance(raw: Any) -> tuple[BalanceSheet, ...]:
    if isinstance(raw, BaseException) or not isinstance(raw, list):
        return ()
    out: list[BalanceSheet] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        out.append(
            BalanceSheet(
                period=_period_from(entry.get("period")),
                fiscal_date=_parse_date(entry.get("date")),
                total_assets=_safe_float(entry.get("totalAssets")),
                total_liabilities=_safe_float(entry.get("totalLiabilities")),
                total_equity=_safe_float(entry.get("totalStockholdersEquity")),
                cash_and_equivalents=_safe_float(entry.get("cashAndCashEquivalents")),
                short_term_investments=_safe_float(entry.get("shortTermInvestments")),
                total_debt=_safe_float(entry.get("totalDebt")),
                long_term_debt=_safe_float(entry.get("longTermDebt")),
                shares_outstanding=_safe_float(
                    entry.get("commonStock") or entry.get("sharesOutstanding")
                ),
                reported_currency=entry.get("reportedCurrency") or "USD",
            )
        )
    return tuple(out)


def _map_cashflow(raw: Any) -> tuple[CashFlow, ...]:
    if isinstance(raw, BaseException) or not isinstance(raw, list):
        return ()
    out: list[CashFlow] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        out.append(
            CashFlow(
                period=_period_from(entry.get("period")),
                fiscal_date=_parse_date(entry.get("date")),
                operating_cash_flow=_safe_float(entry.get("operatingCashFlow")),
                capital_expenditure=_safe_float(entry.get("capitalExpenditure")),
                free_cash_flow=_safe_float(entry.get("freeCashFlow")),
                dividends_paid=_safe_float(entry.get("dividendsPaid")),
                stock_repurchased=_safe_float(entry.get("commonStockRepurchased")),
                reported_currency=entry.get("reportedCurrency") or "USD",
            )
        )
    return tuple(out)


def _map_ratios(raw: Any) -> KeyRatios:
    record = _first_record(raw)
    if record is None:
        return KeyRatios()
    return KeyRatios(
        pe_ratio=_safe_float(record.get("peRatioTTM")),
        forward_pe=None,
        peg_ratio=_safe_float(record.get("pegRatioTTM")),
        price_to_sales=_safe_float(record.get("priceToSalesRatioTTM")),
        price_to_book=_safe_float(record.get("pbRatioTTM")),
        ev_to_ebitda=_safe_float(record.get("enterpriseValueOverEBITDATTM")),
        ev_to_revenue=_safe_float(record.get("evToSalesTTM")),
        debt_to_equity=_safe_float(record.get("debtToEquityTTM")),
        current_ratio=_safe_float(record.get("currentRatioTTM")),
        quick_ratio=_safe_float(record.get("quickRatioTTM")),
        return_on_equity=_safe_float(record.get("roeTTM")),
        return_on_assets=_safe_float(record.get("returnOnTangibleAssetsTTM")),
        gross_margin=None,
        operating_margin=None,
        net_margin=_safe_float(record.get("netProfitMarginTTM")),
        dividend_yield=_safe_float(record.get("dividendYieldTTM")),
        payout_ratio=_safe_float(record.get("payoutRatioTTM")),
        beta=None,
    )


def _map_targets(raw: Any) -> AnalystTargets | None:
    record = _first_record(raw)
    if record is None:
        return None
    targets = AnalystTargets(
        target_high=_safe_float(record.get("targetHigh")),
        target_low=_safe_float(record.get("targetLow")),
        target_mean=_safe_float(record.get("targetConsensus")),
        target_median=_safe_float(record.get("targetMedian")),
        number_of_analysts=_safe_int(record.get("numberOfAnalystsOpinions")),
        recommendation_mean=None,
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


def _first_record(raw: Any) -> dict[str, Any] | None:
    """FMP returns single-record endpoints as ``[{...}]`` arrays."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    if isinstance(raw, dict):
        return raw
    return None


def _period_from(value: Any) -> ReportPeriod:
    """FMP marks annuals as ``"FY"`` and quarterlies as ``"Q1".."Q4"``."""
    if isinstance(value, str) and value.upper().startswith("Q"):
        return ReportPeriod.QUARTERLY
    return ReportPeriod.ANNUAL


def _parse_date(value: Any) -> datetime:
    """FMP dates are ISO ``YYYY-MM-DD``. Fall back to epoch on garbage."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=UTC)


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


__all__ = ["FmpProvider"]
