"""Finnhub provider mapper tests."""

from __future__ import annotations

import pytest

from src.intelligence.fundamentals.finnhub import (
    _map_profile,
    _map_ratios,
    _map_targets,
)


def test_map_profile_pulls_canonical_fields() -> None:
    raw = {
        "name": "Microsoft Corporation",
        "finnhubIndustry": "Software",
        "exchange": "NASDAQ",
        "country": "US",
        "currency": "USD",
        "marketCapitalization": 3_100_000,
        "shareOutstanding": 7_400_000,
    }
    profile = _map_profile("MSFT", raw)
    assert profile.name == "Microsoft Corporation"
    assert profile.sector == "Software"
    assert profile.exchange == "NASDAQ"
    assert profile.market_cap == 3_100_000


def test_map_ratios_uses_metric_subkey() -> None:
    raw = {
        "metric": {
            "peTTM": 30.0,
            "psTTM": 11.0,
            "pegRatio": 2.5,
            "roeTTM": 0.34,
            "beta": 0.95,
        }
    }
    ratios = _map_ratios(raw)
    assert ratios.pe_ratio == pytest.approx(30.0)
    assert ratios.price_to_sales == pytest.approx(11.0)
    assert ratios.peg_ratio == pytest.approx(2.5)
    assert ratios.return_on_equity == pytest.approx(0.34)
    assert ratios.beta == pytest.approx(0.95)


def test_map_ratios_with_no_metric_block_is_empty() -> None:
    ratios = _map_ratios({})
    assert ratios.pe_ratio is None
    assert ratios.beta is None


def test_map_targets_requires_lastupdated_to_be_truthy() -> None:
    assert _map_targets({}) is None
    assert _map_targets({"lastUpdated": "", "targetMean": 100}) is None


def test_map_targets_populates_when_lastupdated_set() -> None:
    targets = _map_targets(
        {
            "lastUpdated": "2026-04-15",
            "targetHigh": 500,
            "targetLow": 350,
            "targetMean": 425,
            "targetMedian": 420,
            "numberOfAnalysts": 22,
        }
    )
    assert targets is not None
    assert targets.target_high == 500
    assert targets.number_of_analysts == 22
