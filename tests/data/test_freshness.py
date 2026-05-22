"""Unit tests for the freshness policy table + evaluator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.data.freshness import (
    POLICIES,
    FreshnessPolicy,
    UnknownPolicyError,
    evaluate_freshness,
    policy_for,
)


def test_policy_table_has_every_canonical_key() -> None:
    """Guard against an entry being accidentally removed from POLICIES.

    The keys are referenced by string from consumer code; a missing
    entry should fail at test time, not at runtime.
    """
    expected_keys = {
        "fundamentals.quarterly",
        "fundamentals.annual",
        "bars.intraday",
        "bars.daily",
        "news",
        "analyst_targets",
    }
    assert set(POLICIES.keys()) >= expected_keys


def test_all_policies_have_strictly_increasing_windows() -> None:
    """fresh < aging < expired is a structural invariant."""
    for key, policy in POLICIES.items():
        assert policy.fresh_window < policy.aging_window, key
        assert policy.aging_window < policy.expired_window, key


def test_misordered_windows_raise_at_construction() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        FreshnessPolicy(
            fresh_window=timedelta(days=10),
            aging_window=timedelta(days=5),  # out of order
            expired_window=timedelta(days=20),
        )


def test_policy_for_unknown_key_raises() -> None:
    with pytest.raises(UnknownPolicyError):
        policy_for("nonsense.key")


def test_policy_for_returns_canonical_entry() -> None:
    assert policy_for("fundamentals.quarterly") is POLICIES["fundamentals.quarterly"]


@pytest.fixture
def policy() -> FreshnessPolicy:
    return FreshnessPolicy(
        fresh_window=timedelta(days=10),
        aging_window=timedelta(days=20),
        expired_window=timedelta(days=30),
    )


def _at(days_ago: float, *, now: datetime) -> datetime:
    return now - timedelta(days=days_ago)


def test_fresh_window_boundary_inclusive(policy: FreshnessPolicy) -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    # Exactly at the fresh boundary stays fresh.
    assert evaluate_freshness(_at(10, now=now), now, policy) == "fresh"


def test_aging_window_boundary(policy: FreshnessPolicy) -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    # One day past fresh → aging.
    assert evaluate_freshness(_at(11, now=now), now, policy) == "aging"
    # Exactly at aging boundary stays aging.
    assert evaluate_freshness(_at(20, now=now), now, policy) == "aging"


def test_stale_window_boundary(policy: FreshnessPolicy) -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    # Past aging boundary → stale.
    assert evaluate_freshness(_at(21, now=now), now, policy) == "stale"
    assert evaluate_freshness(_at(30, now=now), now, policy) == "stale"


def test_expired_beyond_expired_window(policy: FreshnessPolicy) -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    assert evaluate_freshness(_at(31, now=now), now, policy) == "expired"
    assert evaluate_freshness(_at(365, now=now), now, policy) == "expired"


def test_future_as_of_is_clamped_to_fresh(policy: FreshnessPolicy) -> None:
    """Provider clock skew should not flip a future timestamp to expired."""
    now = datetime(2026, 5, 21, tzinfo=UTC)
    future = now + timedelta(hours=1)
    assert evaluate_freshness(future, now, policy) == "fresh"


def test_quarterly_fundamentals_at_90_days_is_fresh() -> None:
    """Concrete check: a Q3 statement filed 90 days ago is still fresh."""
    now = datetime(2026, 5, 21, tzinfo=UTC)
    as_of = now - timedelta(days=90)
    assert evaluate_freshness(as_of, now, policy_for("fundamentals.quarterly")) == "fresh"


def test_quarterly_fundamentals_at_one_year_is_expired() -> None:
    now = datetime(2026, 5, 21, tzinfo=UTC)
    as_of = now - timedelta(days=420)
    assert (
        evaluate_freshness(as_of, now, policy_for("fundamentals.quarterly")) == "expired"
    )


def test_intraday_bars_at_two_minutes_is_aging() -> None:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    as_of = now - timedelta(minutes=2)
    assert evaluate_freshness(as_of, now, policy_for("bars.intraday")) == "aging"


def test_news_at_eight_hours_is_aging() -> None:
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    as_of = now - timedelta(hours=8)
    assert evaluate_freshness(as_of, now, policy_for("news")) == "aging"
