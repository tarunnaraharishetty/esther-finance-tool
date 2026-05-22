"""Tests for the provider accuracy ledger.

Covers the lazy-init contract, write path, rolling-window aggregation,
and the cold-start ``has_events`` probe. Same SQLite/WAL invariants
as :class:`HealthStore`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data.accuracy_store import AccuracyStore
from src.intelligence.fundamentals.models import AccuracyEvent, ProviderName


# ---------------------------------------------------------------------------
# Construction + lazy init
# ---------------------------------------------------------------------------


def test_construction_does_not_touch_disk(tmp_path: Path) -> None:
    """Critical invariant: making the store is free.

    Tests that never exercise the store can construct one safely on a
    bogus path. Same contract HealthStore offers.
    """
    store = AccuracyStore(tmp_path / "does_not_exist" / "accuracy.db")
    assert not (tmp_path / "does_not_exist").exists()
    store.close()


def test_record_creates_db_and_schema_on_first_call(tmp_path: Path) -> None:
    db = tmp_path / "ledger.db"
    store = AccuracyStore(db)
    store.record([_event()])
    assert db.exists()
    # WAL journaling produces sidecar files; mere existence of the main
    # file proves the schema script ran.
    summary = store.summarize(ProviderName.FMP, timedelta(days=30))
    assert summary.total_events == 1
    store.close()


# ---------------------------------------------------------------------------
# Record + summarize
# ---------------------------------------------------------------------------


def test_record_returns_count_written(tmp_path: Path) -> None:
    store = AccuracyStore(tmp_path / "ledger.db")
    n = store.record([_event(), _event(field="net_income")])
    assert n == 2
    store.close()


def test_record_empty_iterable_is_zero_no_db_touch(tmp_path: Path) -> None:
    """Skipping the write when there's nothing to write keeps the DB
    file from materializing on noop calls."""
    store = AccuracyStore(tmp_path / "ledger.db")
    assert store.record([]) == 0
    assert not (tmp_path / "ledger.db").exists()
    store.close()


def test_summarize_with_no_events_returns_zero_totals(tmp_path: Path) -> None:
    """The store always returns a summary — never raises on empty input.

    This is the cold-start contract the trust-weight module depends on.
    """
    store = AccuracyStore(tmp_path / "ledger.db")
    summary = store.summarize(ProviderName.FMP, timedelta(days=90))
    assert summary.total_events == 0
    assert summary.total_agreed == 0
    assert summary.overall_accuracy == 0.0
    assert summary.by_field == ()
    store.close()


def test_summarize_aggregates_per_field(tmp_path: Path) -> None:
    """Each field gets its own ``FieldAccuracy`` row in ``by_field``."""
    store = AccuracyStore(tmp_path / "ledger.db")
    now = datetime.now(UTC)
    store.record(
        [
            _event(field="revenue", agreed=True, observed_at=now),
            _event(field="revenue", agreed=True, observed_at=now),
            _event(field="revenue", agreed=False, observed_at=now),
            _event(field="net_income", agreed=True, observed_at=now),
        ]
    )
    summary = store.summarize(
        ProviderName.FMP, timedelta(days=30), now=now + timedelta(hours=1)
    )
    by_field = {f.field: f for f in summary.by_field}
    assert by_field["revenue"].total == 3
    assert by_field["revenue"].agreed == 2
    assert by_field["revenue"].accuracy == pytest.approx(2 / 3)
    assert by_field["net_income"].total == 1
    assert by_field["net_income"].accuracy == 1.0
    # Overall = total_agreed / total_events.
    assert summary.overall_accuracy == pytest.approx(3 / 4)
    store.close()


def test_summarize_excludes_events_outside_window(tmp_path: Path) -> None:
    """Window cutoff is strict — older events don't pollute the rate."""
    store = AccuracyStore(tmp_path / "ledger.db")
    now = datetime.now(UTC)
    old = now - timedelta(days=200)
    store.record(
        [
            _event(agreed=True, observed_at=now),
            _event(agreed=False, observed_at=old),
            _event(agreed=False, observed_at=old),
        ]
    )
    summary = store.summarize(
        ProviderName.FMP, timedelta(days=30), now=now + timedelta(hours=1)
    )
    assert summary.total_events == 1
    assert summary.overall_accuracy == 1.0
    store.close()


def test_summarize_rejects_non_positive_window(tmp_path: Path) -> None:
    store = AccuracyStore(tmp_path / "ledger.db")
    with pytest.raises(ValueError):
        store.summarize(ProviderName.FMP, timedelta(0))
    store.close()


# ---------------------------------------------------------------------------
# has_events probe
# ---------------------------------------------------------------------------


def test_has_events_false_on_empty_store(tmp_path: Path) -> None:
    store = AccuracyStore(tmp_path / "ledger.db")
    assert store.has_events(ProviderName.FMP) is False
    store.close()


def test_has_events_true_when_provider_appears_as_either_side(
    tmp_path: Path,
) -> None:
    """FMP-as-observed and FMP-as-reference both count as 'has history'."""
    store = AccuracyStore(tmp_path / "ledger.db")
    # FMP is the observed side once, the reference side once.
    store.record(
        [
            _event(
                provider=ProviderName.FMP,
                reference=ProviderName.FINNHUB,
            ),
        ]
    )
    assert store.has_events(ProviderName.FMP) is True
    assert store.has_events(ProviderName.FINNHUB) is True
    assert store.has_events(ProviderName.YAHOO) is False
    store.close()


def test_records_persist_across_store_instances(tmp_path: Path) -> None:
    """Writes are durable — closing and reopening sees the data."""
    db = tmp_path / "ledger.db"
    store_a = AccuracyStore(db)
    store_a.record([_event()])
    store_a.close()

    store_b = AccuracyStore(db)
    summary = store_b.summarize(ProviderName.FMP, timedelta(days=30))
    assert summary.total_events == 1
    store_b.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    provider: ProviderName = ProviderName.FMP,
    reference: ProviderName = ProviderName.SEC_EDGAR,
    symbol: str = "AAPL",
    field: str = "revenue",
    observed_value: float = 100.0,
    reference_value: float = 100.0,
    rel_error: float = 0.0,
    agreed: bool = True,
    fiscal_date: datetime | None = None,
    observed_at: datetime | None = None,
) -> AccuracyEvent:
    return AccuracyEvent(
        provider=provider,
        reference_provider=reference,
        symbol=symbol,
        field=field,
        observed_value=observed_value,
        reference_value=reference_value,
        rel_error=rel_error,
        agreed=agreed,
        fiscal_date=fiscal_date or datetime(2024, 12, 31, tzinfo=UTC),
        observed_at=observed_at or datetime.now(UTC),
    )
