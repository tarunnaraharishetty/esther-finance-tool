"""Unit tests for :class:`DataEnvelope`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from src.data.envelope import DataEnvelope


def _make_envelope(
    *,
    freshness: str = "fresh",
    as_of: datetime | None = None,
    fetched_at: datetime | None = None,
) -> DataEnvelope[str]:
    fetched = fetched_at or datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    asof = as_of or fetched - timedelta(days=10)
    return DataEnvelope(
        data="payload",
        as_of=asof,
        fetched_at=fetched,
        source_chain=("fmp",),
        freshness=freshness,  # type: ignore[arg-type]
        provider_confidence=1.0,
    )


def test_envelope_is_frozen() -> None:
    env = _make_envelope()
    with pytest.raises(FrozenInstanceError):
        env.data = "mutated"  # type: ignore[misc]


def test_data_age_is_positive_delta() -> None:
    env = _make_envelope()
    assert env.data_age == timedelta(days=10)


def test_data_age_clamps_negative_to_zero() -> None:
    """Provider clock skew: as_of in the future. Expose as zero age, not negative."""
    fetched = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    env = _make_envelope(as_of=fetched + timedelta(hours=1), fetched_at=fetched)
    assert env.data_age == timedelta(0)


def test_is_fresh_only_when_freshness_is_fresh() -> None:
    assert _make_envelope(freshness="fresh").is_fresh
    for tier in ("aging", "stale", "expired"):
        assert not _make_envelope(freshness=tier).is_fresh


def test_is_actionable_covers_fresh_and_aging() -> None:
    assert _make_envelope(freshness="fresh").is_actionable
    assert _make_envelope(freshness="aging").is_actionable
    assert not _make_envelope(freshness="stale").is_actionable
    assert not _make_envelope(freshness="expired").is_actionable


def test_envelope_carries_source_chain_in_order() -> None:
    env = DataEnvelope(
        data={"x": 1},
        as_of=datetime(2026, 5, 21, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
        source_chain=("fmp", "finnhub", "yahoo"),
        freshness="aging",
        provider_confidence=0.7,
    )
    assert env.source_chain == ("fmp", "finnhub", "yahoo")
    assert env.data == {"x": 1}
