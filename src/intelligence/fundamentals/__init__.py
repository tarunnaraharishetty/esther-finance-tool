"""Fundamentals provider abstraction (Phase 1).

A pluggable layer that fetches company fundamentals from multiple
providers and normalizes their wildly different response shapes into a
single internal dataclass tree. The rest of the analyzer pipeline
(valuation models, AI explanations, frontend) never sees a raw provider
JSON — everything goes through :class:`NormalizedFundamentals`.

The provider chain is intentional: keyed APIs first (FMP, Finnhub,
Alpha Vantage), authoritative-but-slow second (SEC EDGAR), scraped
last (Yahoo). The orchestrator
(:class:`~src.intelligence.fundamentals.service.FundamentalsService`)
walks the chain, falls back on transient / rate-limit / not-found
errors, and records per-provider health so we can detect a provider
silently rotting.
"""

from src.intelligence.fundamentals.base import (
    FundamentalsProvider,
    ProviderError,
    ProviderNotFound,
    ProviderRateLimited,
    ProviderTransient,
    ProviderUnavailable,
)
from src.intelligence.fundamentals.models import (
    AnalystTargets,
    BalanceSheet,
    CashFlow,
    CompanyProfile,
    IncomeStatement,
    KeyRatios,
    NormalizedFundamentals,
    ProviderHealth,
    ProviderName,
    ProviderRawResponse,
    ReportPeriod,
)
from src.intelligence.fundamentals.service import (
    FundamentalsResult,
    FundamentalsService,
    ProviderChainExhausted,
)

__all__ = [
    "AnalystTargets",
    "BalanceSheet",
    "CashFlow",
    "CompanyProfile",
    "FundamentalsProvider",
    "FundamentalsResult",
    "FundamentalsService",
    "IncomeStatement",
    "KeyRatios",
    "NormalizedFundamentals",
    "ProviderChainExhausted",
    "ProviderError",
    "ProviderHealth",
    "ProviderName",
    "ProviderNotFound",
    "ProviderRateLimited",
    "ProviderRawResponse",
    "ProviderTransient",
    "ProviderUnavailable",
    "ReportPeriod",
]
