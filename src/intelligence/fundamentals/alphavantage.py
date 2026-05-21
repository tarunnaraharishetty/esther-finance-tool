"""Alpha Vantage provider.

Alpha Vantage exposes one endpoint per resource via the ``function``
query parameter:

* ``OVERVIEW``                 — profile + TTM ratios in one blob
* ``INCOME_STATEMENT``         — annual + quarterly arrays
* ``BALANCE_SHEET``
* ``CASH_FLOW``

Free-tier quota is brutal (5 calls / minute, 500 / day) so this
provider should usually sit behind FMP / Finnhub in the chain and act
as a fallback rather than the primary. The orchestrator's per-call
fallback handles that — Alpha Vantage gets tried only if the higher-
priority providers fail or aren't configured.

Alpha Vantage signals quota exhaustion with a 200 OK body containing
``{"Information": "...rate limit..."}`` rather than a 429. We detect
the magic key and raise :class:`ProviderRateLimited` so the chain
falls through.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

from src.config import Settings, get_settings
from src.intelligence.fundamentals.base import (
    ProviderNotFound,
    ProviderRateLimited,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.http import get_json
from src.intelligence.fundamentals.models import (
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

_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider:
    """Alpha Vantage adapter."""

    name = ProviderName.ALPHA_VANTAGE

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
        return self._settings.alphavantage_api_key is not None

    async def fetch(
        self, symbol: str
    ) -> tuple[NormalizedFundamentals, ProviderRawResponse]:
        if not self.is_configured:
            raise ProviderUnavailable(self.name, "ALPHAVANTAGE_API_KEY not set")
        sym = symbol.upper()
        api_key = self._settings.alphavantage_api_key.get_secret_value()  # type: ignore[union-attr]

        async def _get(function: str) -> Any:
            data = await get_json(
                self.name,
                self._base_url,
                params={"function": function, "symbol": sym, "apikey": api_key},
                client=self._client,
            )
            _raise_for_quota(self.name, data)
            return data

        overview_raw, income_raw, balance_raw, cashflow_raw = await asyncio.gather(
            _get("OVERVIEW"),
            _get("INCOME_STATEMENT"),
            _get("BALANCE_SHEET"),
            _get("CASH_FLOW"),
            return_exceptions=True,
        )

        if isinstance(overview_raw, ProviderNotFound):
            raise overview_raw
        if isinstance(overview_raw, BaseException):
            raise overview_raw

        profile = _map_profile(sym, overview_raw)
        ratios = _map_ratios(overview_raw)
        income = _map_income(income_raw)
        balance = _map_balance(balance_raw)
        cashflow = _map_cashflow(cashflow_raw)

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
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=sym,
            endpoint="OVERVIEW,INCOME_STATEMENT,BALANCE_SHEET,CASH_FLOW",
            fetched_at=fetched_at,
            payload={
                "overview": _coerce_payload(overview_raw),
                "income_statement": _coerce_payload(income_raw),
                "balance_sheet": _coerce_payload(balance_raw),
                "cash_flow": _coerce_payload(cashflow_raw),
            },
        )
        return normalized, raw


def _raise_for_quota(provider: ProviderName, data: Any) -> None:
    """Alpha Vantage signals quota errors with a 200 OK message body."""
    if not isinstance(data, dict):
        return
    if "Note" in data or "Information" in data:
        text = data.get("Note") or data.get("Information") or ""
        if "rate limit" in text.lower() or "premium" in text.lower():
            raise ProviderRateLimited(provider, text)


def _coerce_payload(value: Any) -> Any:
    if isinstance(value, BaseException):
        return {"_error": type(value).__name__, "message": str(value)}
    return value


def _map_profile(symbol: str, raw: Any) -> CompanyProfile:
    if not isinstance(raw, dict):
        return CompanyProfile(symbol=symbol)
    return CompanyProfile(
        symbol=symbol,
        name=raw.get("Name"),
        sector=raw.get("Sector"),
        industry=raw.get("Industry"),
        exchange=raw.get("Exchange"),
        country=raw.get("Country"),
        currency=raw.get("Currency") or "USD",
        market_cap=_safe_float(raw.get("MarketCapitalization")),
        shares_outstanding=_safe_float(raw.get("SharesOutstanding")),
        description=raw.get("Description"),
        cik=raw.get("CIK"),
    )


def _map_ratios(raw: Any) -> KeyRatios:
    if not isinstance(raw, dict):
        return KeyRatios()
    return KeyRatios(
        pe_ratio=_safe_float(raw.get("PERatio")),
        forward_pe=_safe_float(raw.get("ForwardPE")),
        peg_ratio=_safe_float(raw.get("PEGRatio")),
        price_to_sales=_safe_float(raw.get("PriceToSalesRatioTTM")),
        price_to_book=_safe_float(raw.get("PriceToBookRatio")),
        ev_to_ebitda=_safe_float(raw.get("EVToEBITDA")),
        ev_to_revenue=_safe_float(raw.get("EVToRevenue")),
        debt_to_equity=None,
        current_ratio=None,
        quick_ratio=None,
        return_on_equity=_safe_float(raw.get("ReturnOnEquityTTM")),
        return_on_assets=_safe_float(raw.get("ReturnOnAssetsTTM")),
        gross_margin=_safe_float(raw.get("GrossProfitTTM")),
        operating_margin=_safe_float(raw.get("OperatingMarginTTM")),
        net_margin=_safe_float(raw.get("ProfitMargin")),
        dividend_yield=_safe_float(raw.get("DividendYield")),
        payout_ratio=_safe_float(raw.get("PayoutRatio")),
        beta=_safe_float(raw.get("Beta")),
    )


def _map_income(raw: Any) -> tuple[IncomeStatement, ...]:
    if not isinstance(raw, dict):
        return ()
    entries = raw.get("annualReports") or []
    out: list[IncomeStatement] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append(
            IncomeStatement(
                period=ReportPeriod.ANNUAL,
                fiscal_date=_parse_date(entry.get("fiscalDateEnding")),
                revenue=_safe_float(entry.get("totalRevenue")),
                cost_of_revenue=_safe_float(entry.get("costOfRevenue")),
                gross_profit=_safe_float(entry.get("grossProfit")),
                operating_income=_safe_float(entry.get("operatingIncome")),
                ebitda=_safe_float(entry.get("ebitda")),
                ebit=_safe_float(entry.get("ebit")),
                net_income=_safe_float(entry.get("netIncome")),
                eps_basic=None,
                eps_diluted=None,
                shares_diluted=None,
                reported_currency=entry.get("reportedCurrency") or "USD",
            )
        )
    return tuple(out)


def _map_balance(raw: Any) -> tuple[BalanceSheet, ...]:
    if not isinstance(raw, dict):
        return ()
    entries = raw.get("annualReports") or []
    out: list[BalanceSheet] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append(
            BalanceSheet(
                period=ReportPeriod.ANNUAL,
                fiscal_date=_parse_date(entry.get("fiscalDateEnding")),
                total_assets=_safe_float(entry.get("totalAssets")),
                total_liabilities=_safe_float(entry.get("totalLiabilities")),
                total_equity=_safe_float(entry.get("totalShareholderEquity")),
                cash_and_equivalents=_safe_float(
                    entry.get("cashAndCashEquivalentsAtCarryingValue")
                ),
                short_term_investments=_safe_float(entry.get("shortTermInvestments")),
                total_debt=_safe_float(entry.get("shortLongTermDebtTotal")),
                long_term_debt=_safe_float(entry.get("longTermDebt")),
                shares_outstanding=_safe_float(entry.get("commonStockSharesOutstanding")),
                reported_currency=entry.get("reportedCurrency") or "USD",
            )
        )
    return tuple(out)


def _map_cashflow(raw: Any) -> tuple[CashFlow, ...]:
    if not isinstance(raw, dict):
        return ()
    entries = raw.get("annualReports") or []
    out: list[CashFlow] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append(
            CashFlow(
                period=ReportPeriod.ANNUAL,
                fiscal_date=_parse_date(entry.get("fiscalDateEnding")),
                operating_cash_flow=_safe_float(entry.get("operatingCashflow")),
                capital_expenditure=_safe_float(entry.get("capitalExpenditures")),
                free_cash_flow=None,
                dividends_paid=_safe_float(entry.get("dividendPayout")),
                stock_repurchased=_safe_float(
                    entry.get("paymentsForRepurchaseOfCommonStock")
                ),
                reported_currency=entry.get("reportedCurrency") or "USD",
            )
        )
    return tuple(out)


def _parse_date(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=UTC)


def _safe_float(value: Any) -> float | None:
    if value is None or value in ("", "None", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["AlphaVantageProvider"]
