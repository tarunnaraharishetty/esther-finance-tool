"""Normalized data model for fundamentals.

Every provider returns its own shape — FMP wraps statements in
``{"symbol": ..., "data": [...]}`` arrays, Finnhub keys metrics under
``metric.metricType``, Alpha Vantage paginates by report kind, SEC
EDGAR returns XBRL concepts. The orchestrator coerces all of them to
the dataclasses in this module so consumers (valuation models, AI
prompts, frontend) only need to know one shape.

Design rules
------------
* Every numeric field is ``float | None``. ``None`` means the provider
  did not supply the value (or it was non-numeric); downstream models
  must handle missing inputs explicitly rather than treating absent as
  zero.
* All money values are in USD. Providers that return reporting
  currency tag their statements with ``reported_currency``; the
  orchestrator records but does not convert (Phase 1 is US equities
  only).
* ``provider`` and ``fetched_at`` live on every record so we can audit
  later which data point came from where.
* Dataclasses are ``frozen=True`` — once normalized, nothing should
  mutate. Use ``dataclasses.replace`` if you need a derived copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ReportPeriod(StrEnum):
    """Reporting cadence of a statement."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    TTM = "ttm"


class ProviderName(StrEnum):
    """Stable identifiers for the providers we know about.

    Used in DB rows, health logs, and the provider chain order setting.
    Keep these short and lowercase — they appear in JSON payloads.
    """

    FMP = "fmp"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    SEC_EDGAR = "sec_edgar"
    YAHOO = "yahoo"


@dataclass(frozen=True)
class IncomeStatement:
    """Top-line through net income for one fiscal period.

    Field names mirror the canonical US-GAAP labels so the SEC EDGAR
    adapter can fill them directly. Other providers map their column
    headers into these names.
    """

    period: ReportPeriod
    fiscal_date: datetime
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    ebitda: float | None = None
    ebit: float | None = None
    net_income: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    shares_diluted: float | None = None
    reported_currency: str = "USD"


@dataclass(frozen=True)
class BalanceSheet:
    period: ReportPeriod
    fiscal_date: datetime
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    total_debt: float | None = None
    long_term_debt: float | None = None
    shares_outstanding: float | None = None
    reported_currency: str = "USD"


@dataclass(frozen=True)
class CashFlow:
    period: ReportPeriod
    fiscal_date: datetime
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    dividends_paid: float | None = None
    stock_repurchased: float | None = None
    reported_currency: str = "USD"


@dataclass(frozen=True)
class KeyRatios:
    """TTM ratios computed by the provider (or by us at normalization time).

    These are the inputs to the multiples-based valuation models in
    Phase 4 — keeping them on the normalized object so each model
    doesn't have to recompute from raw statements.
    """

    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_sales: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_revenue: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    beta: float | None = None


@dataclass(frozen=True)
class AnalystTargets:
    """Street consensus target prices.

    Optional — many providers don't expose this on the free tier, and
    SEC EDGAR never does. The valuation ensemble treats absence as
    "this model contributes nothing this run" rather than crashing.
    """

    target_high: float | None = None
    target_low: float | None = None
    target_mean: float | None = None
    target_median: float | None = None
    number_of_analysts: int | None = None
    recommendation_mean: float | None = None  # 1=Strong Buy ... 5=Strong Sell


@dataclass(frozen=True)
class CompanyProfile:
    """Identity + sector classification used for sector-median lookups."""

    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    country: str | None = None
    currency: str = "USD"
    market_cap: float | None = None
    enterprise_value: float | None = None
    shares_outstanding: float | None = None
    description: str | None = None
    cik: str | None = None  # SEC central index key — needed for EDGAR lookups


@dataclass(frozen=True)
class ProviderRawResponse:
    """One raw payload from one provider, ready to persist.

    Stored verbatim under ``provider_raw_responses`` so we can re-parse
    history with a new mapper without re-hitting the upstream API. The
    ``payload`` is whatever JSON the provider returned (parsed); for
    SEC EDGAR it's the decoded XBRL companyfacts dict.
    """

    provider: ProviderName
    symbol: str
    endpoint: str
    fetched_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderHealth:
    """One row for the provider health log.

    Recorded once per orchestrator call, per provider attempted.
    Status is the outcome from the orchestrator's perspective:

    * ``ok`` — provider returned a usable response.
    * ``empty`` — 200 OK but no data for the symbol (e.g. unlisted).
    * ``rate_limited`` — 429 / quota error.
    * ``unavailable`` — 5xx / network / API key missing.
    * ``transient`` — temporary failure worth retrying next call.
    """

    provider: ProviderName
    symbol: str
    status: str
    latency_ms: float
    checked_at: datetime
    error_message: str | None = None


@dataclass(frozen=True)
class NormalizedFundamentals:
    """The full normalized record for one symbol.

    A single instance can be assembled from multiple providers — e.g.
    company profile from FMP, statements from SEC EDGAR, analyst
    targets from Finnhub — by passing the same object through the
    chain with ``replace``. In Phase 1 we typically just take the first
    healthy provider's full payload; the merge logic is a Phase 2
    refinement.
    """

    symbol: str
    fetched_at: datetime
    primary_provider: ProviderName
    contributing_providers: tuple[ProviderName, ...]
    profile: CompanyProfile
    income_statements: tuple[IncomeStatement, ...] = ()
    balance_sheets: tuple[BalanceSheet, ...] = ()
    cash_flows: tuple[CashFlow, ...] = ()
    key_ratios: KeyRatios = field(default_factory=KeyRatios)
    analyst_targets: AnalystTargets | None = None
    warnings: tuple[str, ...] = ()

    @property
    def latest_annual_income(self) -> IncomeStatement | None:
        """Most recent annual income statement, or ``None`` if absent."""
        annuals = [s for s in self.income_statements if s.period is ReportPeriod.ANNUAL]
        return max(annuals, key=lambda s: s.fiscal_date) if annuals else None

    @property
    def latest_annual_balance(self) -> BalanceSheet | None:
        annuals = [s for s in self.balance_sheets if s.period is ReportPeriod.ANNUAL]
        return max(annuals, key=lambda s: s.fiscal_date) if annuals else None

    @property
    def latest_annual_cashflow(self) -> CashFlow | None:
        annuals = [s for s in self.cash_flows if s.period is ReportPeriod.ANNUAL]
        return max(annuals, key=lambda s: s.fiscal_date) if annuals else None


__all__ = [
    "AnalystTargets",
    "BalanceSheet",
    "CashFlow",
    "CompanyProfile",
    "IncomeStatement",
    "KeyRatios",
    "NormalizedFundamentals",
    "ProviderHealth",
    "ProviderName",
    "ProviderRawResponse",
    "ReportPeriod",
]
