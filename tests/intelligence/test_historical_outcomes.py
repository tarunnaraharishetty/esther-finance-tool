"""Unit tests for :func:`symbol_outcomes`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.intelligence.calibration import CalibrationStore
from src.intelligence.historical_outcomes import (
    HistoricalOutcomes,
    SymbolBucket,
    outcomes_to_wire,
    symbol_outcomes,
)


def _store(tmp_path: Path) -> CalibrationStore:
    return CalibrationStore(tmp_path / "calibration.db")


def _seed(
    store: CalibrationStore,
    *,
    symbol: str,
    score_name: str,
    outcome_name: str,
    score_value: float,
    n: int,
    hits: int,
    horizon_days: int = 5,
    base: datetime | None = None,
) -> None:
    moment = base or datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n):
        oid = store.record_observation(
            symbol=symbol,
            score_name=score_name,
            score_value=score_value,
            observed_at=moment + timedelta(hours=i),
            horizon_days=horizon_days,
            outcome_name=outcome_name,
            starting_price=100.0,
        )
        store.record_outcome(oid, i < hits)


# -----------------------------------------------------------------------------
# symbol_outcomes — empty + populated
# -----------------------------------------------------------------------------


def test_empty_store_returns_zero_counts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    assert isinstance(outcomes, HistoricalOutcomes)
    assert outcomes.symbol == "AAPL"
    assert outcomes.total_observations == 0
    assert outcomes.settled_observations == 0
    assert outcomes.buckets == ()
    assert outcomes.first_observed_at is None
    assert outcomes.last_settled_at is None


def test_uppercases_symbol(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=10,
        hits=7,
    )
    outcomes = symbol_outcomes(store, "aapl", horizon_days=5)
    assert outcomes.symbol == "AAPL"
    assert outcomes.settled_observations == 10


def test_groups_observations_into_buckets(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=12,
        hits=8,
    )
    outcomes = symbol_outcomes(
        store, "AAPL", horizon_days=5, min_observations=5
    )
    assert len(outcomes.buckets) == 1
    bucket = outcomes.buckets[0]
    assert bucket.bucket_lo == 70.0
    assert bucket.bucket_hi == 80.0
    assert bucket.n_observations == 12
    assert bucket.n_hits == 8
    assert bucket.hit_rate == pytest.approx(8 / 12, abs=1e-6)
    assert bucket.bucket_published is True
    assert 0 <= bucket.confidence_low < bucket.hit_rate < bucket.confidence_high <= 1


def test_under_sampled_bucket_marked_unpublished(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=3,  # below default min_observations=5
        hits=2,
    )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    assert outcomes.buckets[0].bucket_published is False
    # Hit rate + CI still computed — only the publish flag changes.
    assert outcomes.buckets[0].hit_rate == pytest.approx(2 / 3, abs=1e-6)


def test_min_observations_override(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=3,
        hits=2,
    )
    # Lower the floor → bucket publishes.
    outcomes = symbol_outcomes(
        store, "AAPL", horizon_days=5, min_observations=2
    )
    assert outcomes.buckets[0].bucket_published is True


def test_filters_to_symbol(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=10,
        hits=6,
    )
    _seed(
        store,
        symbol="MSFT",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=10,
        hits=9,
    )
    aapl = symbol_outcomes(store, "AAPL", horizon_days=5)
    msft = symbol_outcomes(store, "MSFT", horizon_days=5)
    assert aapl.buckets[0].n_hits == 6
    assert msft.buckets[0].n_hits == 9


def test_filters_to_horizon(tmp_path: Path) -> None:
    """Observations at horizon=10 must not surface in a horizon=5 query."""
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=10,
        hits=7,
        horizon_days=10,
    )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    assert outcomes.settled_observations == 0
    assert outcomes.buckets == ()


def test_unsettled_excluded_from_buckets_but_counted_in_total(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    # 5 settled + 3 pending.
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=5,
        hits=3,
    )
    for i in range(3):
        store.record_observation(
            symbol="AAPL",
            score_name="pullback_risk",
            score_value=75.0,
            observed_at=datetime.now(UTC) + timedelta(seconds=i),
            horizon_days=5,
            outcome_name="return_negative",
            starting_price=100.0,
        )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    assert outcomes.total_observations == 8
    assert outcomes.settled_observations == 5
    assert outcomes.buckets[0].n_observations == 5


def test_groups_by_score_and_outcome(tmp_path: Path) -> None:
    """Different score/outcome combinations land in separate buckets."""
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=10,
        hits=7,
    )
    _seed(
        store,
        symbol="AAPL",
        score_name="rebound_potential",
        outcome_name="return_positive",
        score_value=15.0,
        n=10,
        hits=5,
        base=datetime(2026, 2, 1, tzinfo=UTC),
    )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    by_score = {(b.score_name, b.outcome_name) for b in outcomes.buckets}
    assert ("pullback_risk", "return_negative") in by_score
    assert ("rebound_potential", "return_positive") in by_score


def test_top_bucket_inclusive_at_100(tmp_path: Path) -> None:
    """Score exactly 100 must land in the [90, 100] bucket."""
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=100.0,
        n=5,
        hits=4,
    )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    assert outcomes.buckets[0].bucket_lo == 90.0
    assert outcomes.buckets[0].bucket_hi == 100.0


def test_first_observed_and_last_settled_timestamps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=5,
        hits=3,
        base=base,
    )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    assert outcomes.first_observed_at == base
    # Last settled is the latest observed_at across settled rows.
    assert outcomes.last_settled_at == base + timedelta(hours=4)


# -----------------------------------------------------------------------------
# Argument validation
# -----------------------------------------------------------------------------


def test_invalid_bucket_width_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="bucket_width"):
        symbol_outcomes(store, "AAPL", horizon_days=5, bucket_width=0)
    with pytest.raises(ValueError, match="bucket_width"):
        symbol_outcomes(store, "AAPL", horizon_days=5, bucket_width=150)


def test_invalid_min_observations_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="min_observations"):
        symbol_outcomes(store, "AAPL", horizon_days=5, min_observations=0)


# -----------------------------------------------------------------------------
# Wire shape
# -----------------------------------------------------------------------------


def test_wire_shape_includes_all_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(
        store,
        symbol="AAPL",
        score_name="pullback_risk",
        outcome_name="return_negative",
        score_value=75.0,
        n=10,
        hits=7,
    )
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    payload = outcomes_to_wire(outcomes)

    expected_top_keys = {
        "symbol",
        "horizon_days",
        "total_observations",
        "settled_observations",
        "first_observed_at",
        "last_settled_at",
        "buckets",
    }
    assert set(payload.keys()) == expected_top_keys

    bucket_keys = {
        "score_name",
        "outcome_name",
        "horizon_days",
        "bucket_lo",
        "bucket_hi",
        "n_observations",
        "n_hits",
        "hit_rate",
        "confidence_low",
        "confidence_high",
        "bucket_published",
    }
    assert set(payload["buckets"][0].keys()) == bucket_keys


def test_wire_shape_renders_empty_buckets_when_no_observations(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    outcomes = symbol_outcomes(store, "AAPL", horizon_days=5)
    payload = outcomes_to_wire(outcomes)
    assert payload["buckets"] == []
    assert payload["first_observed_at"] is None
    assert payload["last_settled_at"] is None


# Silence the SymbolBucket import while preserving the type contract.
_ = SymbolBucket
