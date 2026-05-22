"""Tests for the analyzer-side calibration bridge."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.intelligence.analyzer.calibration import (
    PAIRINGS,
    CalibrationReading,
    lookup_readings,
)
from src.intelligence.analyzer.technical import (
    TechnicalScores,
    TechnicalSubscores,
)
from src.intelligence.calibration import CalibrationStore


def _subscores(**overrides: float | None) -> TechnicalSubscores:
    base: dict[str, float | None] = {
        "rsi_overbought_score": None,
        "rsi_oversold_score": None,
        "bollinger_extension_score": None,
        "ma_extension_score": None,
        "volume_spike_score": None,
        "atr_volatility_score": None,
        "momentum_exhaustion_score": None,
    }
    base.update(overrides)
    return TechnicalSubscores(**base)  # type: ignore[arg-type]


def _technicals(
    *,
    overbought: float | None = None,
    oversold: float | None = None,
    pullback: float | None = None,
    rebound: float | None = None,
) -> TechnicalScores:
    return TechnicalScores(
        subscores=_subscores(),
        overbought_score=overbought,
        oversold_score=oversold,
        pullback_risk=pullback,
        rebound_potential=rebound,
        confidence_score=50.0,
        raw_rsi=None,
        raw_bollinger_z=None,
        raw_ma_distance_pct=None,
        raw_volume_z=None,
        raw_atr_ratio=None,
    )


def _seed_bucket(
    store: CalibrationStore,
    *,
    score_name: str,
    score_value: float,
    n: int,
    hits: int,
    outcome_name: str,
    horizon_days: int = 5,
) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n):
        oid = store.record_observation(
            symbol=f"S{i % 5:02d}",
            score_name=score_name,
            score_value=score_value,
            observed_at=base + timedelta(hours=i),
            horizon_days=horizon_days,
            outcome_name=outcome_name,
        )
        store.record_outcome(oid, i < hits)


def _build_published_table(tmp_path: Path) -> CalibrationStore:
    """Seed two pre-published buckets for the analyzer's score pairings."""
    store = CalibrationStore(tmp_path / "calibration.db")
    _seed_bucket(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=40,
        outcome_name="return_negative",
    )
    _seed_bucket(
        store,
        score_name="rebound_potential",
        score_value=15.0,
        n=10,  # under-sampled at default min_observations=30
        hits=6,
        outcome_name="return_positive",
    )
    store.build_table(
        score_names=["pullback_risk", "rebound_potential"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    store.build_table(
        score_names=["pullback_risk", "rebound_potential"],
        horizon_days=5,
        outcome_name="return_positive",
    )
    return store


# -----------------------------------------------------------------------------
# Pairings invariant
# -----------------------------------------------------------------------------


def test_pairings_cover_the_documented_score_set() -> None:
    """Adding/removing a pairing should be a deliberate code change.

    Pin the set so a future refactor can't silently drop one.
    """
    expected = {
        ("overbought_score", "return_negative"),
        ("oversold_score", "return_positive"),
        ("pullback_risk", "return_negative"),
        ("rebound_potential", "return_positive"),
    }
    assert set(PAIRINGS) == expected


# -----------------------------------------------------------------------------
# lookup_readings — happy path
# -----------------------------------------------------------------------------


def test_lookup_emits_reading_per_populated_score(tmp_path: Path) -> None:
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(pullback=73.0, rebound=15.0)

    readings = lookup_readings(
        technicals, table, horizon_days=5, min_observations=30
    )
    by_name = {r.score_name: r for r in readings}
    # Both pullback_risk and rebound_potential have score values;
    # the other two pairings drop because the technical fields are None.
    assert "pullback_risk" in by_name
    assert "rebound_potential" in by_name
    assert "overbought_score" not in by_name


def test_lookup_marks_well_sampled_bucket_as_published(
    tmp_path: Path,
) -> None:
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(pullback=73.0)

    readings = lookup_readings(
        technicals, table, horizon_days=5, min_observations=30
    )
    reading = next(r for r in readings if r.score_name == "pullback_risk")
    assert reading.bucket_published is True
    assert reading.bucket is not None
    assert reading.bucket.hit_rate == pytest.approx(0.80)
    assert reading.bucket.n_observations == 50


def test_lookup_marks_under_sampled_bucket_as_unpublished(
    tmp_path: Path,
) -> None:
    """Rebound bucket only has 10 observations — below default floor."""
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(rebound=15.0)

    readings = lookup_readings(
        technicals, table, horizon_days=5, min_observations=30
    )
    reading = next(r for r in readings if r.score_name == "rebound_potential")
    assert reading.bucket_published is False
    assert reading.bucket is None
    # Score value is still preserved so the UI can show the raw read.
    assert reading.score_value == pytest.approx(15.0)


def test_lookup_preserves_score_value_on_published_reading(
    tmp_path: Path,
) -> None:
    """The exact score_value the analyzer computed must round-trip
    onto the reading — the trader's question is "what does *this*
    setup imply", not "what was the bucket midpoint"."""
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(pullback=73.4)

    reading = next(
        r
        for r in lookup_readings(
            technicals, table, horizon_days=5, min_observations=30
        )
        if r.score_name == "pullback_risk"
    )
    assert reading.score_value == pytest.approx(73.4)


# -----------------------------------------------------------------------------
# lookup_readings — None / degraded paths
# -----------------------------------------------------------------------------


def test_lookup_returns_empty_when_table_missing(tmp_path: Path) -> None:
    technicals = _technicals(pullback=73.0)
    assert lookup_readings(technicals, None, horizon_days=5, min_observations=30) == ()


def test_lookup_returns_empty_when_technicals_missing(tmp_path: Path) -> None:
    store = _build_published_table(tmp_path)
    table = store.load_table()
    assert lookup_readings(None, table, horizon_days=5, min_observations=30) == ()


def test_lookup_skips_pairings_with_none_score(tmp_path: Path) -> None:
    """A pairing whose score wasn't computable doesn't emit a reading.

    No "the analyzer didn't produce this score" ghost rows.
    """
    store = _build_published_table(tmp_path)
    table = store.load_table()
    # Only pullback set; overbought/oversold/rebound are None.
    technicals = _technicals(pullback=73.0)

    readings = lookup_readings(
        technicals, table, horizon_days=5, min_observations=30
    )
    assert {r.score_name for r in readings} == {"pullback_risk"}


def test_lookup_respects_min_observations_override(tmp_path: Path) -> None:
    """Lowering the floor lets the under-sampled bucket publish.

    Tests the kill-switch path: an operator running with looser
    sampling rules (e.g., bootstrapping a new score) should be able
    to see provisional probabilities without modifying any code.
    """
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(rebound=15.0)

    readings = lookup_readings(
        technicals, table, horizon_days=5, min_observations=5
    )
    reading = next(r for r in readings if r.score_name == "rebound_potential")
    assert reading.bucket_published is True
    assert reading.bucket is not None
    assert reading.bucket.n_observations == 10


def test_lookup_respects_horizon_filter(tmp_path: Path) -> None:
    """A different horizon means a different bucket set — no cross-pollination."""
    store = CalibrationStore(tmp_path / "calibration.db")
    # Build a 5-day bucket; ask for a 10-day reading → no match → unpublished.
    _seed_bucket(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=40,
        outcome_name="return_negative",
    )
    store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    table = store.load_table()
    technicals = _technicals(pullback=73.0)

    readings = lookup_readings(
        technicals, table, horizon_days=10, min_observations=30
    )
    reading = next(r for r in readings if r.score_name == "pullback_risk")
    assert reading.bucket_published is False


# -----------------------------------------------------------------------------
# Frozen dataclass invariants
# -----------------------------------------------------------------------------


def test_reading_is_frozen(tmp_path: Path) -> None:
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(pullback=73.0)
    reading = lookup_readings(
        technicals, table, horizon_days=5, min_observations=30
    )[0]
    with pytest.raises(Exception):
        reading.score_value = 999.0  # type: ignore[misc]


def test_reading_dataclass_shape(tmp_path: Path) -> None:
    """Pin the field set so a future field addition is a deliberate change."""
    store = _build_published_table(tmp_path)
    table = store.load_table()
    technicals = _technicals(pullback=73.0)
    reading = lookup_readings(
        technicals, table, horizon_days=5, min_observations=30
    )[0]
    assert isinstance(reading, CalibrationReading)
    assert reading.score_name == "pullback_risk"
    assert reading.outcome_name == "return_negative"
    assert reading.horizon_days == 5
