"""Tests for the pure sector-ranking module.

Covers per-metric rank correctness for both directions, percentile
math, cohort grouping by sector alias, singleton/missing-sector
edge cases, tied-rank assignment, and the wire shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.intelligence.sector_ranker import (
    SECTOR_RANK_METRICS,
    SymbolRanking,
    compute_sector_rankings,
)


def _report(
    symbol: str,
    sector: str | None = "Information Technology",
    *,
    trust: float | None = 90.0,
    last_price: float = 100.0,
    overbought: float | None = 40.0,
    oversold: float | None = 60.0,
    pullback_risk: float | None = 30.0,
    rebound_potential: float | None = 70.0,
    provider_conf: float | None = 0.9,
    valuation_conf: float | None = 75.0,
    base_case: float | None = 120.0,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "last_price": last_price,
        "fundamentals_profile": (
            {"sector": sector} if sector is not None else None
        ),
        "trust_score": (
            {"score": trust, "grade": "A", "components": []}
            if trust is not None
            else None
        ),
        "fundamentals_freshness": (
            {"provider_confidence": provider_conf}
            if provider_conf is not None
            else None
        ),
        "valuation": (
            {"confidence_score": valuation_conf, "base_case": base_case}
            if valuation_conf is not None or base_case is not None
            else None
        ),
        "technicals": {
            "overbought_score": overbought,
            "oversold_score": oversold,
            "pullback_risk": pullback_risk,
            "rebound_potential": rebound_potential,
        },
    }


def _by_metric(ranking: SymbolRanking) -> dict[str, Any]:
    return {m.metric: m for m in ranking.metrics}


# ---------------------------------------------------------------------------
# Spec sanity
# ---------------------------------------------------------------------------


def test_spec_carries_eight_metrics() -> None:
    """Lock the metric count — adding/removing requires a UI update."""
    assert len(SECTOR_RANK_METRICS) == 8
    assert {s.metric for s in SECTOR_RANK_METRICS} == {
        "trust_score",
        "provider_confidence",
        "valuation_confidence",
        "upside_to_base_case",
        "overbought_score",
        "oversold_score",
        "pullback_risk",
        "rebound_potential",
    }


# ---------------------------------------------------------------------------
# Cohort grouping + ranking
# ---------------------------------------------------------------------------


def test_higher_better_metric_assigns_rank_1_to_max() -> None:
    """Trust score is higher_better → top score gets rank=1."""
    reports = [
        _report("AAA", trust=90),
        _report("BBB", trust=75),
        _report("CCC", trust=50),
    ]
    rankings = compute_sector_rankings(reports)
    assert _by_metric(rankings["AAA"])["trust_score"].rank == 1
    assert _by_metric(rankings["BBB"])["trust_score"].rank == 2
    assert _by_metric(rankings["CCC"])["trust_score"].rank == 3


def test_lower_better_metric_assigns_rank_1_to_min() -> None:
    """Overbought is lower_better → least overbought gets rank=1."""
    reports = [
        _report("AAA", overbought=80),
        _report("BBB", overbought=50),
        _report("CCC", overbought=20),
    ]
    rankings = compute_sector_rankings(reports)
    assert _by_metric(rankings["CCC"])["overbought_score"].rank == 1
    assert _by_metric(rankings["BBB"])["overbought_score"].rank == 2
    assert _by_metric(rankings["AAA"])["overbought_score"].rank == 3


def test_percentile_formula_maps_top_to_100_bottom_to_0() -> None:
    """Rank 1 of N → 100; rank N of N → 0; linear in between."""
    reports = [
        _report(s, trust=t)
        for s, t in [("AAA", 100), ("BBB", 75), ("CCC", 50), ("DDD", 25)]
    ]
    rankings = compute_sector_rankings(reports)
    assert _by_metric(rankings["AAA"])["trust_score"].percentile == pytest.approx(
        100.0
    )
    assert _by_metric(rankings["DDD"])["trust_score"].percentile == pytest.approx(
        0.0
    )
    # Rank 2 of 4 → (4-2)/(4-1) * 100 ≈ 66.67
    assert _by_metric(rankings["BBB"])["trust_score"].percentile == pytest.approx(
        66.6667, abs=1e-3
    )


def test_ties_share_rank_and_subsequent_ranks_skip() -> None:
    """Three-way tie at rank 1 → next symbol gets rank 4 (standard skip)."""
    reports = [
        _report("AAA", trust=90),
        _report("BBB", trust=90),
        _report("CCC", trust=90),
        _report("DDD", trust=80),
    ]
    rankings = compute_sector_rankings(reports)
    ranks = sorted(
        _by_metric(rankings[s])["trust_score"].rank or 0
        for s in ("AAA", "BBB", "CCC")
    )
    assert ranks == [1, 1, 1]
    assert _by_metric(rankings["DDD"])["trust_score"].rank == 4


def test_derived_upside_metric_uses_base_case_and_last_price() -> None:
    """upside_to_base_case = (base - last) / last * 100, ranked higher_better."""
    reports = [
        _report("AAA", last_price=100, base_case=150),  # +50%
        _report("BBB", last_price=100, base_case=110),  # +10%
        _report("CCC", last_price=100, base_case=80),   # -20%
    ]
    rankings = compute_sector_rankings(reports)
    upside = {
        s: _by_metric(rankings[s])["upside_to_base_case"]
        for s in ("AAA", "BBB", "CCC")
    }
    assert upside["AAA"].value == pytest.approx(50.0)
    assert upside["AAA"].rank == 1
    assert upside["CCC"].value == pytest.approx(-20.0)
    assert upside["CCC"].rank == 3


def test_missing_value_excludes_symbol_from_metric_cohort() -> None:
    """A symbol with no value gets rank=None and shrinks the cohort_size."""
    reports = [
        _report("AAA", trust=90),
        _report("BBB", trust=None),  # missing trust
        _report("CCC", trust=70),
    ]
    rankings = compute_sector_rankings(reports)
    bbb = _by_metric(rankings["BBB"])["trust_score"]
    assert bbb.rank is None
    assert bbb.percentile is None
    assert bbb.value is None
    aaa = _by_metric(rankings["AAA"])["trust_score"]
    assert aaa.rank == 1
    assert aaa.cohort_size == 2  # AAA + CCC; BBB excluded


# ---------------------------------------------------------------------------
# Sector grouping + edge cases
# ---------------------------------------------------------------------------


def test_sector_aliases_collapse_to_same_cohort() -> None:
    """'Tech' and 'Information Technology' must land in the same cohort."""
    reports = [
        _report("AAA", sector="Information Technology", trust=90),
        _report("BBB", sector="Tech", trust=80),
        _report("CCC", sector="technology", trust=70),
    ]
    rankings = compute_sector_rankings(reports)
    assert rankings["AAA"].cohort_size == 3
    assert rankings["BBB"].cohort_size == 3
    assert rankings["CCC"].cohort_size == 3


def test_singleton_cohort_returns_degraded_ranking() -> None:
    """A symbol with no sector peers returns an empty ranking, not a fake #1."""
    reports = [
        _report("AAA", sector="Information Technology"),
        _report("BBB", sector="Financials"),
        _report("CCC", sector="Health Care"),
    ]
    rankings = compute_sector_rankings(reports)
    for sym in ("AAA", "BBB", "CCC"):
        assert rankings[sym].cohort_size == 0
        assert rankings[sym].metrics == ()


def test_symbol_with_unknown_sector_gets_degraded_ranking() -> None:
    """An empty/None sector excludes the symbol from grouping."""
    reports = [
        _report("AAA", sector=None),
        _report("BBB", sector="Information Technology"),
        _report("CCC", sector="Information Technology"),
    ]
    rankings = compute_sector_rankings(reports)
    assert rankings["AAA"].cohort_size == 0
    assert rankings["AAA"].sector is None
    assert rankings["BBB"].cohort_size == 2
    assert rankings["CCC"].cohort_size == 2


def test_cohort_symbols_is_sorted_alphabetic() -> None:
    """``cohort_symbols`` is deterministic for stable UI display."""
    reports = [
        _report("ZZZ"),
        _report("AAA"),
        _report("MMM"),
    ]
    rankings = compute_sector_rankings(reports)
    for sym in ("AAA", "MMM", "ZZZ"):
        assert rankings[sym].cohort_symbols == ("AAA", "MMM", "ZZZ")


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_shape() -> None:
    """The wire dict carries the exact fields the TS types expect."""
    reports = [_report("AAA"), _report("BBB", trust=70)]
    rankings = compute_sector_rankings(reports)
    wire = rankings["AAA"].to_dict()

    assert set(wire.keys()) == {
        "symbol",
        "sector",
        "cohort_size",
        "cohort_symbols",
        "metrics",
    }
    assert isinstance(wire["metrics"], list)
    for metric in wire["metrics"]:
        assert set(metric.keys()) == {
            "metric",
            "label",
            "value",
            "rank",
            "cohort_size",
            "percentile",
            "direction",
            "percent",
        }
        assert metric["direction"] in {"higher_better", "lower_better"}


def test_percentile_rounded_to_one_decimal_in_wire() -> None:
    """``to_dict`` rounds percentile to keep the wire size tame."""
    reports = [
        _report("AAA", trust=90),
        _report("BBB", trust=80),
        _report("CCC", trust=70),
    ]
    rankings = compute_sector_rankings(reports)
    wire_metric = next(
        m for m in rankings["BBB"].to_dict()["metrics"]
        if m["metric"] == "trust_score"
    )
    # Rank 2 of 3 → (3-2)/(3-1) * 100 = 50.0
    assert wire_metric["percentile"] == 50.0


def test_two_symbol_cohort_still_ranks() -> None:
    """The smallest usable cohort (2 symbols) produces ranks 1 and 2."""
    reports = [
        _report("AAA", trust=90),
        _report("BBB", trust=75),
    ]
    rankings = compute_sector_rankings(reports)
    aaa_trust = _by_metric(rankings["AAA"])["trust_score"]
    bbb_trust = _by_metric(rankings["BBB"])["trust_score"]
    assert aaa_trust.rank == 1
    assert bbb_trust.rank == 2
    # Percentile of 2-symbol cohort: rank 1 → 100, rank 2 → 0.
    assert aaa_trust.percentile == pytest.approx(100.0)
    assert bbb_trust.percentile == pytest.approx(0.0)


def test_empty_report_list_returns_empty_dict() -> None:
    assert compute_sector_rankings([]) == {}
