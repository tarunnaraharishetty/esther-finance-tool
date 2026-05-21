"""SEC EDGAR provider.

EDGAR is the authoritative US source — the same XBRL filings every
other paid provider derives from. No API key required, but the SEC
mandates a contact User-Agent string on every request. We hit two
endpoints:

* ``https://www.sec.gov/files/company_tickers.json`` — symbol -> CIK map,
  fetched once and cached in-process.
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK<10-digit>.json`` —
  every reported XBRL concept across every filing for the company.

XBRL is verbose. We pull the GAAP concepts the analyzer actually
uses (revenue, net income, total assets, etc.), pick the most recent
10-K ``USD`` value for each, and map them onto a single annual
:class:`IncomeStatement` / :class:`BalanceSheet` / :class:`CashFlow`.
Quarterly history can be added in a follow-up — for Phase 1 we just
need a viable fallback when paid providers are down.
"""

from __future__ import annotations

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
    BalanceSheet,
    CashFlow,
    CompanyProfile,
    IncomeStatement,
    NormalizedFundamentals,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# GAAP concept names → normalized field. EDGAR sometimes uses several
# legitimate aliases for the same idea (revenue is the worst); we try
# each in order and take the first that yields a value.
_INCOME_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "eps_basic": ("EarningsPerShareBasic",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
}

_BALANCE_CONCEPTS: dict[str, tuple[str, ...]] = {
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "total_equity": ("StockholdersEquity",),
    "cash_and_equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
    "short_term_investments": ("ShortTermInvestments",),
    "long_term_debt": ("LongTermDebt",),
    "total_debt": ("LongTermDebt",),  # rough — refine in Phase 2
    "shares_outstanding": ("CommonStockSharesOutstanding",),
}

_CASHFLOW_CONCEPTS: dict[str, tuple[str, ...]] = {
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "stock_repurchased": ("PaymentsForRepurchaseOfCommonStock",),
}


class SecEdgarProvider:
    """SEC EDGAR XBRL companyfacts adapter."""

    name = ProviderName.SEC_EDGAR

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._ticker_cache: dict[str, str] | None = None

    @property
    def is_configured(self) -> bool:
        # EDGAR requires a contact User-Agent, not a key. We treat an
        # empty UA as misconfigured rather than silently sending the
        # default httpx UA (the SEC blocks generic clients).
        return bool(self._settings.sec_edgar_user_agent.strip())

    async def fetch(
        self, symbol: str
    ) -> tuple[NormalizedFundamentals, ProviderRawResponse]:
        if not self.is_configured:
            raise ProviderUnavailable(self.name, "SEC_EDGAR_USER_AGENT not set")
        sym = symbol.upper()
        headers = {"User-Agent": self._settings.sec_edgar_user_agent}

        cik = await self._lookup_cik(sym, headers)
        if cik is None:
            raise ProviderNotFound(self.name, f"no CIK match for {sym}")

        facts = await get_json(
            self.name,
            _COMPANYFACTS_URL.format(cik=cik),
            headers=headers,
            client=self._client,
        )
        gaap = _get_gaap(facts)

        income = _build_income(gaap)
        balance = _build_balance(gaap)
        cashflow = _build_cashflow(gaap)
        profile = CompanyProfile(
            symbol=sym,
            name=_safe_str(facts.get("entityName")) if isinstance(facts, dict) else None,
            cik=cik,
        )

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
            warnings=("SEC EDGAR provides statements only; sector / market cap "
                      "/ ratios must come from another provider.",),
        )
        raw = ProviderRawResponse(
            provider=self.name,
            symbol=sym,
            endpoint=f"companyfacts/CIK{cik}",
            fetched_at=fetched_at,
            payload=facts if isinstance(facts, dict) else {"_raw": facts},
        )
        return normalized, raw

    async def _lookup_cik(self, symbol: str, headers: dict[str, str]) -> str | None:
        if self._ticker_cache is None:
            data = await get_json(
                self.name, _TICKERS_URL, headers=headers, client=self._client
            )
            mapping: dict[str, str] = {}
            # The SEC ticker file is shaped as
            # ``{"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}``.
            if isinstance(data, dict):
                for entry in data.values():
                    if not isinstance(entry, dict):
                        continue
                    ticker = entry.get("ticker")
                    cik = entry.get("cik_str")
                    if isinstance(ticker, str) and cik is not None:
                        mapping[ticker.upper()] = f"{int(cik):010d}"
            self._ticker_cache = mapping
        return self._ticker_cache.get(symbol)


def _get_gaap(facts: Any) -> dict[str, Any]:
    if not isinstance(facts, dict):
        return {}
    facts_block = facts.get("facts")
    if not isinstance(facts_block, dict):
        return {}
    gaap = facts_block.get("us-gaap")
    return gaap if isinstance(gaap, dict) else {}


def _latest_annual_usd(
    gaap: dict[str, Any], aliases: tuple[str, ...]
) -> tuple[float | None, datetime | None]:
    """Walk concept aliases; return the most recent FY USD value."""
    for concept in aliases:
        block = gaap.get(concept)
        if not isinstance(block, dict):
            continue
        units = block.get("units")
        if not isinstance(units, dict):
            continue
        # Prefer USD; fall back to USD/shares for per-share metrics.
        usd_entries = units.get("USD") or units.get("USD/shares") or units.get("shares")
        if not isinstance(usd_entries, list):
            continue
        annuals = [
            e
            for e in usd_entries
            if isinstance(e, dict) and e.get("fp") == "FY" and e.get("form") == "10-K"
        ]
        if not annuals:
            continue
        annuals.sort(key=lambda e: e.get("end") or "", reverse=True)
        latest = annuals[0]
        end = _parse_date(latest.get("end"))
        val = latest.get("val")
        try:
            return float(val) if val is not None else None, end
        except (TypeError, ValueError):
            continue
    return None, None


def _build_income(gaap: dict[str, Any]) -> tuple[IncomeStatement, ...]:
    if not gaap:
        return ()
    fields: dict[str, float | None] = {}
    dates: list[datetime] = []
    for field_name, aliases in _INCOME_CONCEPTS.items():
        value, when = _latest_annual_usd(gaap, aliases)
        fields[field_name] = value
        if when is not None:
            dates.append(when)
    if not dates:
        return ()
    fiscal_date = max(dates)
    return (
        IncomeStatement(
            period=ReportPeriod.ANNUAL,
            fiscal_date=fiscal_date,
            **fields,  # type: ignore[arg-type]
        ),
    )


def _build_balance(gaap: dict[str, Any]) -> tuple[BalanceSheet, ...]:
    if not gaap:
        return ()
    fields: dict[str, float | None] = {}
    dates: list[datetime] = []
    for field_name, aliases in _BALANCE_CONCEPTS.items():
        value, when = _latest_annual_usd(gaap, aliases)
        fields[field_name] = value
        if when is not None:
            dates.append(when)
    if not dates:
        return ()
    fiscal_date = max(dates)
    return (
        BalanceSheet(
            period=ReportPeriod.ANNUAL,
            fiscal_date=fiscal_date,
            **fields,  # type: ignore[arg-type]
        ),
    )


def _build_cashflow(gaap: dict[str, Any]) -> tuple[CashFlow, ...]:
    if not gaap:
        return ()
    fields: dict[str, float | None] = {}
    dates: list[datetime] = []
    for field_name, aliases in _CASHFLOW_CONCEPTS.items():
        value, when = _latest_annual_usd(gaap, aliases)
        fields[field_name] = value
        if when is not None:
            dates.append(when)
    if not dates:
        return ()
    # Free cash flow = OCF - CapEx when both are present.
    ocf = fields.get("operating_cash_flow")
    capex = fields.get("capital_expenditure")
    fcf = ocf - capex if ocf is not None and capex is not None else None
    fields["free_cash_flow"] = fcf
    fiscal_date = max(dates)
    return (
        CashFlow(
            period=ReportPeriod.ANNUAL,
            fiscal_date=fiscal_date,
            **fields,  # type: ignore[arg-type]
        ),
    )


def _parse_date(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=UTC)


def _safe_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["SecEdgarProvider"]
