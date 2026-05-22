"""Unit tests for :mod:`src.intelligence.calibration`."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.intelligence.calibration import (
    CalibrationBucket,
    CalibrationStore,
    CalibrationTable,
    wilson_interval,
)


def _store(tmp_path: Path) -> CalibrationStore:
    return CalibrationStore(tmp_path / "calibration.db")


def _seed_bucket(
    store: CalibrationStore,
    *,
    score_name: str,
    score_value: float,
    n: int,
    hits: int,
    horizon_days: int = 5,
    outcome_name: str = "return_negative",
    base_observed: datetime | None = None,
) -> None:
    """Seed ``n`` observations at one score_value, with ``hits`` outcomes True."""
    base = base_observed or datetime(2026, 1, 1, tzinfo=UTC)
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


# -----------------------------------------------------------------------------
# Construction + lazy init
# -----------------------------------------------------------------------------


def test_init_does_not_touch_disk(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "calibration.db"
    CalibrationStore(path)
    assert not path.exists()
    assert not path.parent.exists()


def test_first_write_creates_db_and_parent(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path / "nested" / "calibration.db")
    store.record_observation(
        symbol="AAPL",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
    )
    assert store.db_path.exists()


def test_record_observation_rejects_non_positive_horizon(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        store.record_observation(
            symbol="AAPL",
            score_name="pullback_risk",
            score_value=73.0,
            observed_at=datetime.now(UTC),
            horizon_days=0,
            outcome_name="return_negative",
        )


# -----------------------------------------------------------------------------
# Observation round-trip
# -----------------------------------------------------------------------------


def test_observation_round_trips_through_disk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    oid = store.record_observation(
        symbol="aapl",  # lowercase → store should uppercase
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=when,
        horizon_days=5,
        outcome_name="return_negative",
    )
    rows = store.all_observations()
    assert len(rows) == 1
    obs = rows[0]
    assert obs.id == oid
    assert obs.symbol == "AAPL"
    assert obs.score_value == 73.0
    assert obs.observed_at == when
    assert obs.horizon_days == 5
    assert obs.outcome_value is None
    assert obs.matures_at == when + timedelta(days=5)


def test_record_outcome_sets_value_and_only_for_matching_row(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    a = store.record_observation(
        symbol="AAPL",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
    )
    b = store.record_observation(
        symbol="MSFT",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=datetime.now(UTC),
        horizon_days=5,
        outcome_name="return_negative",
    )
    assert store.record_outcome(a, True) is True
    rows = {o.id: o for o in store.all_observations()}
    assert rows[a].outcome_value is True
    assert rows[b].outcome_value is None


def test_record_outcome_on_unknown_id_returns_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.record_outcome(9999, True) is False


def test_matured_unsettled_filters_by_horizon(tmp_path: Path) -> None:
    """Only observations whose maturation date has passed surface."""
    store = _store(tmp_path)
    early = datetime(2026, 1, 1, tzinfo=UTC)
    store.record_observation(
        symbol="OLD",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=early,
        horizon_days=5,
        outcome_name="return_negative",
    )
    store.record_observation(
        symbol="NEW",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=early + timedelta(days=30),
        horizon_days=5,
        outcome_name="return_negative",
    )
    matured = store.matured_unsettled(now=early + timedelta(days=10))
    # OLD matured at day 5; NEW matures at day 35 → only OLD returned.
    assert {m.symbol for m in matured} == {"OLD"}


def test_matured_excludes_settled(tmp_path: Path) -> None:
    store = _store(tmp_path)
    oid = store.record_observation(
        symbol="AAPL",
        score_name="pullback_risk",
        score_value=73.0,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        horizon_days=5,
        outcome_name="return_negative",
    )
    store.record_outcome(oid, True)
    matured = store.matured_unsettled(now=datetime(2026, 6, 1, tzinfo=UTC))
    assert matured == ()


# -----------------------------------------------------------------------------
# Wilson interval math
# -----------------------------------------------------------------------------


def testwilson_interval_known_value() -> None:
    """64 hits in 100 trials → ~54.2–72.7% per Wilson 95%.

    Pinned via hand calc with z=1.959963984540054 (the precise 97.5th
    standard-normal quantile). A textbook with z=1.96 lands in the
    same ballpark to two decimal places.
    """
    lo, hi = wilson_interval(64, 100)
    assert lo == pytest.approx(0.5424, abs=1e-3)
    assert hi == pytest.approx(0.7273, abs=1e-3)


def testwilson_interval_at_zero_hits() -> None:
    """p_hat = 0 must not divide by zero. Upper bound is positive."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


def testwilson_interval_at_full_hits() -> None:
    lo, hi = wilson_interval(10, 10)
    assert 0.0 < lo < 1.0
    # ``min(1.0, …)`` clamps at the boundary; float precision may
    # land at 1.0 - eps before the clamp. ``pytest.approx`` covers
    # the eps without softening the invariant.
    assert hi == pytest.approx(1.0)


def testwilson_interval_zero_observations() -> None:
    """n = 0 → no information → [0, 1] (uninformative)."""
    lo, hi = wilson_interval(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def testwilson_interval_is_asymmetric_at_small_n() -> None:
    """A defensible property: at small n with p_hat near boundary,
    the Wilson interval is asymmetric — the boundary-near side is
    tighter than the boundary-far side."""
    lo, hi = wilson_interval(1, 5)  # p_hat = 0.2 with very small n
    center = (lo + hi) / 2
    # Center should be pulled toward 0.5, not stuck at 0.2.
    assert center > 0.2


# -----------------------------------------------------------------------------
# build_table: hit rate, bucket boundaries, CI persistence
# -----------------------------------------------------------------------------


def test_build_table_computes_hit_rate_per_bucket(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Two buckets: [70, 80) with 80% hit rate, [80, 90) with 50%.
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=50, hits=40)
    _seed_bucket(
        store,
        score_name="pullback_risk",
        score_value=85.0,
        n=50,
        hits=25,
        base_observed=datetime(2026, 2, 1, tzinfo=UTC),
    )
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    by_lo = {b.bucket_lo: b for b in table.buckets}
    assert by_lo[70.0].n_observations == 50
    assert by_lo[70.0].n_hits == 40
    assert by_lo[70.0].hit_rate == pytest.approx(0.80)
    assert by_lo[80.0].hit_rate == pytest.approx(0.50)


def test_build_table_skips_observations_with_no_outcome(tmp_path: Path) -> None:
    """An unsettled observation must not skew the bucket."""
    store = _store(tmp_path)
    # 10 settled hits at 75 + 10 unsettled at 75 → should be 100% hit
    # rate over n=10, NOT 50% over n=20.
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=10, hits=10)
    for i in range(10):
        store.record_observation(
            symbol="UNSETTLED",
            score_name="pullback_risk",
            score_value=75.0,
            observed_at=datetime(2026, 3, 1, tzinfo=UTC) + timedelta(hours=i),
            horizon_days=5,
            outcome_name="return_negative",
        )
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    by_lo = {b.bucket_lo: b for b in table.buckets}
    assert by_lo[70.0].n_observations == 10
    assert by_lo[70.0].hit_rate == pytest.approx(1.0)


def test_build_table_top_bucket_includes_100(tmp_path: Path) -> None:
    """A perfect 100 must fall into the [90, 100] bucket, not get dropped."""
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=100.0, n=10, hits=5)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    assert len(table.buckets) == 1
    bucket = table.buckets[0]
    assert bucket.bucket_lo == 90.0
    assert bucket.bucket_hi == 100.0
    assert bucket.contains(100.0)


def test_build_table_persists_to_disk(tmp_path: Path) -> None:
    """A second instance on the same path loads the built table."""
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=50, hits=40)
    store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    store.close()

    store2 = CalibrationStore(tmp_path / "calibration.db")
    table = store2.load_table()
    assert len(table.buckets) == 1
    assert table.buckets[0].n_observations == 50


def test_build_table_is_atomic_replace_per_score(tmp_path: Path) -> None:
    """Rebuilding must wipe the prior rows for the same triple."""
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=50, hits=40)
    store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    # Insert more observations and rebuild — should overwrite, not duplicate.
    _seed_bucket(
        store,
        score_name="pullback_risk",
        score_value=75.0,
        n=50,
        hits=30,
        base_observed=datetime(2026, 4, 1, tzinfo=UTC),
    )
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    by_lo = {b.bucket_lo: b for b in table.buckets}
    assert by_lo[70.0].n_observations == 100  # 50 + 50, not 150
    # 40 + 30 hits out of 100.
    assert by_lo[70.0].n_hits == 70


def test_build_table_rejects_invalid_bucket_width(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="bucket_width"):
        store.build_table(
            score_names=["pullback_risk"],
            horizon_days=5,
            outcome_name="return_negative",
            bucket_width=0,
        )
    with pytest.raises(ValueError, match="bucket_width"):
        store.build_table(
            score_names=["pullback_risk"],
            horizon_days=5,
            outcome_name="return_negative",
            bucket_width=150,
        )


# -----------------------------------------------------------------------------
# Lookup
# -----------------------------------------------------------------------------


def test_lookup_returns_correct_bucket(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=50, hits=40)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    bucket = table.lookup(
        "pullback_risk",
        73.0,
        horizon_days=5,
        outcome_name="return_negative",
    )
    assert bucket is not None
    assert bucket.bucket_lo == 70.0
    assert bucket.hit_rate == pytest.approx(0.80)


def test_lookup_under_sampled_returns_none(tmp_path: Path) -> None:
    """Bucket with < min_observations must NOT publish a probability."""
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=10, hits=8)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    # Default min_observations=30 → 10 is below it.
    assert (
        table.lookup(
            "pullback_risk",
            73.0,
            horizon_days=5,
            outcome_name="return_negative",
        )
        is None
    )


def test_lookup_respects_min_observations_override(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=10, hits=8)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    bucket = table.lookup(
        "pullback_risk",
        73.0,
        horizon_days=5,
        outcome_name="return_negative",
        min_observations=5,
    )
    assert bucket is not None
    assert bucket.n_observations == 10


def test_lookup_returns_none_on_unknown_score_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=50, hits=40)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    assert (
        table.lookup(
            "rebound_potential",
            73.0,
            horizon_days=5,
            outcome_name="return_positive",
        )
        is None
    )


def test_lookup_returns_none_when_value_outside_any_bucket(tmp_path: Path) -> None:
    """A score in a bucket we never observed → under-sampled → None."""
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=50, hits=40)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    # We have data for [70, 80) only; ask about [20, 30).
    assert (
        table.lookup(
            "pullback_risk",
            25.0,
            horizon_days=5,
            outcome_name="return_negative",
        )
        is None
    )


def test_bucket_contains_boundary_semantics() -> None:
    """[lo, hi) for non-top buckets; [lo, hi] for the top bucket."""
    mid = CalibrationBucket(
        score_name="pullback_risk",
        bucket_lo=70.0,
        bucket_hi=80.0,
        horizon_days=5,
        outcome_name="return_negative",
        n_observations=50,
        n_hits=40,
        hit_rate=0.8,
        confidence_low=0.66,
        confidence_high=0.89,
        last_updated=datetime.now(UTC),
    )
    assert mid.contains(70.0)
    assert mid.contains(79.999)
    assert not mid.contains(80.0)
    assert not mid.contains(69.999)

    top = CalibrationBucket(
        score_name="pullback_risk",
        bucket_lo=90.0,
        bucket_hi=100.0,
        horizon_days=5,
        outcome_name="return_negative",
        n_observations=50,
        n_hits=40,
        hit_rate=0.8,
        confidence_low=0.66,
        confidence_high=0.89,
        last_updated=datetime.now(UTC),
    )
    assert top.contains(90.0)
    assert top.contains(100.0)
    assert not top.contains(89.999)
    assert not top.contains(100.0001)


# -----------------------------------------------------------------------------
# CalibrationTable convenience
# -----------------------------------------------------------------------------


def test_table_total_observations_is_sum(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=30, hits=20)
    _seed_bucket(
        store,
        score_name="pullback_risk",
        score_value=85.0,
        n=20,
        hits=10,
        base_observed=datetime(2026, 2, 1, tzinfo=UTC),
    )
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    assert table.total_observations == 50


def test_empty_table_lookup_returns_none() -> None:
    table = CalibrationTable(buckets=(), built_at=datetime.now(UTC))
    assert (
        table.lookup(
            "pullback_risk",
            73.0,
            horizon_days=5,
            outcome_name="return_negative",
        )
        is None
    )


# -----------------------------------------------------------------------------
# Confidence interval is reported on the bucket
# -----------------------------------------------------------------------------


def test_confidence_interval_is_populated_on_bucket(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_bucket(store, score_name="pullback_risk", score_value=75.0, n=100, hits=64)
    table = store.build_table(
        score_names=["pullback_risk"],
        horizon_days=5,
        outcome_name="return_negative",
    )
    bucket = table.buckets[0]
    # Same Wilson math as the unit test above.
    assert bucket.confidence_low == pytest.approx(0.5424, abs=1e-3)
    assert bucket.confidence_high == pytest.approx(0.7273, abs=1e-3)
    assert bucket.confidence_low < bucket.hit_rate < bucket.confidence_high


# Silence unused-import warning; math reserved for future tests.
_ = math
