"""SEC EDGAR provider mapper tests.

The HTTP path needs the SEC's ticker-to-CIK map and the XBRL companyfacts
blob — both are fetched lazily in production. Here we exercise the
XBRL builders directly against a hand-written facts fragment so the
concept-aliasing logic stays honest.
"""

from __future__ import annotations

import pytest

from src.intelligence.fundamentals.sec_edgar import (
    _build_balance,
    _build_cashflow,
    _build_income,
    _get_gaap,
    _latest_annual_usd,
)


def _fact(concept: str, *entries: dict[str, object]) -> dict[str, object]:
    return {concept: {"units": {"USD": list(entries)}}}


def _facts(*concept_blocks: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for block in concept_blocks:
        merged.update(block)
    return {"facts": {"us-gaap": merged}}


def test_latest_annual_usd_picks_most_recent_fy_10k() -> None:
    block = _facts(
        _fact(
            "Revenues",
            {"fp": "FY", "form": "10-K", "end": "2023-12-31", "val": 100},
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 130},
            {"fp": "Q3", "form": "10-Q", "end": "2024-09-30", "val": 35},
        )
    )
    gaap = _get_gaap(block)
    value, when = _latest_annual_usd(gaap, ("Revenues",))
    assert value == 130
    assert when is not None
    assert when.year == 2024


def test_latest_annual_usd_walks_aliases() -> None:
    # First alias has no FY data; second alias does — _latest_annual_usd
    # should keep walking and pick up the second.
    block = _facts(
        _fact("Revenues", {"fp": "Q1", "form": "10-Q", "end": "2024-03-31", "val": 25}),
        _fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 100},
        ),
    )
    gaap = _get_gaap(block)
    value, when = _latest_annual_usd(
        gaap,
        ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    )
    assert value == 100
    assert when is not None and when.year == 2024


def test_build_income_returns_one_normalized_statement() -> None:
    block = _facts(
        _fact(
            "Revenues",
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 200},
        ),
        _fact(
            "NetIncomeLoss",
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 40},
        ),
    )
    gaap = _get_gaap(block)
    statements = _build_income(gaap)
    assert len(statements) == 1
    assert statements[0].revenue == 200
    assert statements[0].net_income == 40
    assert statements[0].operating_income is None
    assert statements[0].fiscal_date.year == 2024


def test_build_cashflow_computes_free_cash_flow() -> None:
    block = _facts(
        _fact(
            "NetCashProvidedByUsedInOperatingActivities",
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 100},
        ),
        _fact(
            "PaymentsToAcquirePropertyPlantAndEquipment",
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 30},
        ),
    )
    gaap = _get_gaap(block)
    cashflows = _build_cashflow(gaap)
    assert len(cashflows) == 1
    assert cashflows[0].operating_cash_flow == 100
    assert cashflows[0].capital_expenditure == 30
    assert cashflows[0].free_cash_flow == pytest.approx(70)


def test_build_returns_empty_when_no_concepts_match() -> None:
    # Concepts the builder cares about are absent — we should get
    # empty tuples rather than placeholder statements with all-None
    # values.
    block = _facts(
        _fact(
            "SomeRandomConcept",
            {"fp": "FY", "form": "10-K", "end": "2024-12-31", "val": 1},
        )
    )
    gaap = _get_gaap(block)
    assert _build_income(gaap) == ()
    assert _build_balance(gaap) == ()
    assert _build_cashflow(gaap) == ()


def test_get_gaap_tolerates_missing_facts_structure() -> None:
    assert _get_gaap(None) == {}
    assert _get_gaap({"facts": {}}) == {}
    assert _get_gaap({"facts": {"us-gaap": "garbage"}}) == {}
