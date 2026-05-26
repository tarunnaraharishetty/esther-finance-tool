"""Tests for the cohort-based sector-median compute helper.

First cut of `BUGS.md` B-13 — replaces the static seed for sectors
where the live watchlist has enough coverage. Helper is not yet wired
into the analyzer pipeline (a follow-up will add a refresh worker);
this test pins the math.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.intelligence.analyzer.sector_medians import (
    SectorMultiples,
    compute_medians_from_cohort,
    lookup,
    refresh_computed_medians,
    register_computed_medians,
)
from src.intelligence.fundamentals.models import (
    CompanyProfile,
    KeyRatios,
    NormalizedFundamentals,
    ProviderName,
)


def _record(sector: str, *, pe: float | None, ev_ebitda: float | None = None) -> NormalizedFundamentals:
    return NormalizedFundamentals(
        symbol="X",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="X", sector=sector),
        key_ratios=KeyRatios(pe_ratio=pe, ev_to_ebitda=ev_ebitda),
    )


def test_returns_empty_for_empty_cohort() -> None:
    assert compute_medians_from_cohort([]) == {}


def test_sectorless_records_dropped() -> None:
    records = [_record(sector="", pe=20.0)]
    assert compute_medians_from_cohort(records) == {}


def test_median_computed_when_cohort_meets_floor() -> None:
    # 5 tech records with PEs 10, 20, 30, 40, 50 → median 30.
    records = [_record(sector="Tech", pe=float(p)) for p in (10, 20, 30, 40, 50)]
    out = compute_medians_from_cohort(records)
    assert "information technology" in out
    multiples = out["information technology"]
    assert multiples.pe == 30.0
    assert multiples.source == "computed"
    assert multiples.cohort_size == 5
    assert multiples.as_of is not None


def test_below_floor_falls_back_to_seed() -> None:
    # Only 2 samples for healthcare — below the 5-sample floor.
    # The seed for Health Care from the static table should be used.
    records = [_record(sector="Healthcare", pe=p) for p in (15.0, 18.0)]
    out = compute_medians_from_cohort(records)
    assert "health care" in out
    seed = lookup("Healthcare")
    assert out["health care"].pe == seed.pe
    assert out["health care"].cohort_size == 2
    # Still labelled "computed" — the metadata reflects the cohort
    # even when the value fell back to the seed for this metric.
    assert out["health care"].source == "computed"


def test_negative_and_zero_pe_samples_excluded() -> None:
    # 5 records but two are -5 (no earnings) and one is 0 (undefined).
    # Excluding leaves 2 valid samples → below floor → seed.
    records = [
        _record(sector="Tech", pe=-5.0),
        _record(sector="Tech", pe=0.0),
        _record(sector="Tech", pe=10.0),
        _record(sector="Tech", pe=20.0),
        _record(sector="Tech", pe=-100.0),
    ]
    out = compute_medians_from_cohort(records)
    seed = lookup("Tech")
    assert out["information technology"].pe == seed.pe


def test_sector_aliases_collapse_to_canonical_key() -> None:
    # "Tech", "Technology", and "Information Technology" should all be
    # one cohort.
    records = (
        [_record(sector="Tech", pe=10.0)]
        + [_record(sector="Technology", pe=20.0)]
        + [_record(sector="Information Technology", pe=p) for p in (30.0, 40.0, 50.0)]
    )
    out = compute_medians_from_cohort(records)
    assert list(out.keys()) == ["information technology"]
    assert out["information technology"].pe == 30.0  # median of [10,20,30,40,50]


def test_sector_multiples_dataclass_defaults() -> None:
    """Static-seed entries leave the new metadata fields at defaults."""
    sm = SectorMultiples(pe=20.0, ev_ebitda=13.0, ps=2.5, peg=2.0)
    assert sm.source == "static"
    assert sm.as_of is None
    assert sm.cohort_size == 0


# ---------------------------------------------------------------------------
# lookup() respects the computed override when installed
# ---------------------------------------------------------------------------


def test_lookup_uses_computed_override_when_installed() -> None:
    """An installed override takes precedence over the static seed."""
    register_computed_medians(
        {
            "information technology": SectorMultiples(
                pe=99.0,
                ev_ebitda=88.0,
                ps=7.0,
                peg=4.0,
                source="computed",
                cohort_size=12,
            )
        }
    )
    result = lookup("Tech")
    assert result.pe == 99.0
    assert result.source == "computed"
    assert result.cohort_size == 12


def test_lookup_falls_back_to_static_when_sector_missing_from_override() -> None:
    """A sparse override leaves unrelated sectors on the static seed."""
    register_computed_medians(
        {
            "information technology": SectorMultiples(
                pe=99.0,
                ev_ebitda=88.0,
                ps=7.0,
                peg=4.0,
                source="computed",
                cohort_size=12,
            )
        }
    )
    # Healthcare isn't in the override → falls through to the static seed.
    result = lookup("Healthcare")
    assert result.source == "static"


def test_register_with_none_clears_override() -> None:
    """Passing ``None`` reverts to the static seed for every sector."""
    register_computed_medians(
        {
            "information technology": SectorMultiples(
                pe=99.0,
                ev_ebitda=88.0,
                ps=7.0,
                peg=4.0,
                source="computed",
                cohort_size=12,
            )
        }
    )
    register_computed_medians(None)
    result = lookup("Tech")
    assert result.source == "static"


def test_refresh_computed_medians_installs_table() -> None:
    """End-to-end: compute from records, install, observe via lookup."""
    records = [_record(sector="Tech", pe=float(p)) for p in (10, 20, 30, 40, 50)]
    installed = refresh_computed_medians(records)
    assert installed["information technology"].pe == 30.0
    # lookup() now sees the computed value, not the static seed.
    assert lookup("Tech").pe == 30.0
    assert lookup("Tech").source == "computed"


# ---------------------------------------------------------------------------
# SectorCohort — the production driver that drives lookup() from live fetches
# ---------------------------------------------------------------------------


from src.intelligence.analyzer.sector_medians import SectorCohort  # noqa: E402


def _tech_record(symbol: str, pe: float) -> NormalizedFundamentals:
    return NormalizedFundamentals(
        symbol=symbol,
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol=symbol, sector="Information Technology"),
        key_ratios=KeyRatios(pe_ratio=pe),
    )


def test_cohort_rejects_invalid_max_symbols() -> None:
    import pytest

    with pytest.raises(ValueError, match=">= 1"):
        SectorCohort(max_symbols=0)


def test_cohort_observe_drives_lookup() -> None:
    """Closes the loop B-13 asked for: observing live records makes
    lookup() return source='computed' for the freshly-populated sector."""
    cohort = SectorCohort()
    for i, pe in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
        cohort.observe(_tech_record(f"SYM{i}", pe))
    result = lookup("Tech")
    assert result.source == "computed"
    assert result.pe == 30.0  # median of [10, 20, 30, 40, 50]
    assert result.cohort_size == 5
    assert result.as_of is not None


def test_cohort_per_symbol_latest_wins() -> None:
    """Re-observing the same symbol replaces the prior record rather
    than stacking — the cohort represents the latest view, not a log."""
    cohort = SectorCohort()
    cohort.observe(_tech_record("AAPL", pe=10.0))
    cohort.observe(_tech_record("AAPL", pe=100.0))
    # Only one symbol → below floor → falls back to static seed for PE,
    # but the cohort itself has length 1.
    assert len(cohort) == 1
    assert cohort.snapshot()[0].key_ratios.pe_ratio == 100.0


def test_cohort_fifo_eviction_when_over_cap() -> None:
    """Observing past ``max_symbols`` drops the oldest symbol first."""
    cohort = SectorCohort(max_symbols=3)
    for i in range(5):
        cohort.observe(_tech_record(f"SYM{i}", pe=float(10 + i)))
    snapshot = cohort.snapshot()
    symbols = [r.symbol for r in snapshot]
    # Oldest two (SYM0, SYM1) evicted; latest three retained.
    assert symbols == ["SYM2", "SYM3", "SYM4"]


def test_cohort_re_observe_protects_from_eviction() -> None:
    """A re-observed symbol moves to the newest slot, so a frequently
    refreshed ticker doesn't get evicted under cap pressure."""
    cohort = SectorCohort(max_symbols=3)
    cohort.observe(_tech_record("AAPL", pe=10.0))
    cohort.observe(_tech_record("MSFT", pe=20.0))
    cohort.observe(_tech_record("NVDA", pe=30.0))
    # Re-observe AAPL — it should move to "newest", protecting from eviction.
    cohort.observe(_tech_record("AAPL", pe=11.0))
    # Now add a 4th symbol. The oldest is MSFT (NOT AAPL).
    cohort.observe(_tech_record("GOOGL", pe=40.0))
    symbols = [r.symbol for r in cohort.snapshot()]
    assert "AAPL" in symbols
    assert "MSFT" not in symbols


def test_cohort_below_floor_keeps_static_seed_for_metric() -> None:
    """Three observations isn't enough to override the static PE for
    Tech — the per-metric floor stays at the seed."""
    cohort = SectorCohort()
    for i in range(3):
        cohort.observe(_tech_record(f"SYM{i}", pe=float(i + 10)))
    result = lookup("Tech")
    # Source labelled computed (we did run the cohort) but the PE value
    # itself fell back to the static seed because 3 < _MIN_COHORT_SIZE.
    seed = SectorMultiples(pe=30.0, ev_ebitda=20.0, ps=6.0, peg=1.5)  # see config/sector_medians.toml
    assert result.source == "computed"
    # PE came from the static table since the cohort is sub-floor.
    assert result.pe == seed.pe
    assert result.cohort_size == 3


def test_cohort_unknown_sector_still_falls_back_to_static_default() -> None:
    """Symbols with no sector tag don't contaminate the cohort and the
    unknown-sector lookup still serves the _default block."""
    cohort = SectorCohort()
    # No sector → record is silently dropped from the cohort.
    record = NormalizedFundamentals(
        symbol="UNKWN",
        fetched_at=datetime.now(UTC),
        primary_provider=ProviderName.FMP,
        contributing_providers=(ProviderName.FMP,),
        profile=CompanyProfile(symbol="UNKWN", sector=None),
    )
    cohort.observe(record)
    # No computed override for unknown sectors → static default served.
    result = lookup("MysteryMeat")
    assert result.source == "static"
