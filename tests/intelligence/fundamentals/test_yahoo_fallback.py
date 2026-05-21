"""Yahoo fallback provider mapper tests."""

from __future__ import annotations

import pytest

from src.intelligence.fundamentals.yahoo_fallback import (
    _first_result,
    _map_profile,
    _map_ratios,
    _map_targets,
)


def _result(**modules: object) -> dict[str, object]:
    return {"quoteSummary": {"result": [modules]}}


def test_first_result_unwraps_quote_summary_envelope() -> None:
    payload = _result(price={"longName": "Apple Inc."})
    result = _first_result(payload)
    assert result is not None
    assert result["price"] == {"longName": "Apple Inc."}


def test_first_result_returns_none_on_unexpected_shape() -> None:
    assert _first_result(None) is None
    assert _first_result({"quoteSummary": {}}) is None
    assert _first_result({"quoteSummary": {"result": None}}) is None


def test_map_profile_unwraps_raw_values() -> None:
    result = {
        "price": {"longName": "Apple Inc.", "exchangeName": "NMS", "currency": "USD"},
        "assetProfile": {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "longBusinessSummary": "iPhones.",
        },
        "summaryDetail": {"marketCap": {"raw": 3_000_000_000_000, "fmt": "3T"}},
        "defaultKeyStatistics": {
            "enterpriseValue": {"raw": 3_100_000_000_000},
            "sharesOutstanding": {"raw": 15_000_000_000},
        },
    }
    profile = _map_profile("AAPL", result)
    assert profile.name == "Apple Inc."
    assert profile.sector == "Technology"
    assert profile.market_cap == pytest.approx(3e12)
    assert profile.enterprise_value == pytest.approx(3.1e12)
    assert profile.shares_outstanding == pytest.approx(1.5e10)


def test_map_ratios_pulls_from_multiple_modules() -> None:
    result = {
        "summaryDetail": {
            "trailingPE": {"raw": 30.0},
            "forwardPE": {"raw": 28.0},
            "priceToSalesTrailing12Months": {"raw": 7.5},
            "beta": {"raw": 1.2},
        },
        "defaultKeyStatistics": {
            "pegRatio": {"raw": 2.0},
            "priceToBook": {"raw": 40.0},
            "enterpriseToEbitda": {"raw": 25.0},
        },
        "financialData": {
            "grossMargins": {"raw": 0.46},
            "returnOnEquity": {"raw": 1.5},
        },
    }
    ratios = _map_ratios(result)
    assert ratios.pe_ratio == pytest.approx(30.0)
    assert ratios.forward_pe == pytest.approx(28.0)
    assert ratios.peg_ratio == pytest.approx(2.0)
    assert ratios.beta == pytest.approx(1.2)
    assert ratios.gross_margin == pytest.approx(0.46)


def test_map_targets_returns_none_when_all_missing() -> None:
    assert _map_targets({"financialData": {}}) is None


def test_map_targets_populates_when_present() -> None:
    targets = _map_targets(
        {
            "financialData": {
                "targetHighPrice": {"raw": 250},
                "targetLowPrice": {"raw": 180},
                "targetMeanPrice": {"raw": 215},
                "targetMedianPrice": {"raw": 220},
                "numberOfAnalystOpinions": {"raw": 30},
                "recommendationMean": {"raw": 1.8},
            }
        }
    )
    assert targets is not None
    assert targets.target_high == 250
    assert targets.recommendation_mean == pytest.approx(1.8)
