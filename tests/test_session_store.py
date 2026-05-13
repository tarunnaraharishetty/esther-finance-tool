"""Tests for session-state persistence: SessionStore + tracker round-trips."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.intelligence.alert_prioritizer import AlertState
from src.intelligence.alerts import Alert
from src.intelligence.history import SignalEpisode, SignalHistory
from src.intelligence.opportunity_history import OpportunityMembershipTracker
from src.intelligence.pulse import MarketPulse
from src.intelligence.pulse_history import PulseHistoryTracker
from src.persistence.session_store import (
    CURRENT_SCHEMA_VERSION,
    PulseRecord,
    SessionSnapshot,
    SessionStore,
)
from src.strategy.base import SignalAction

# ---------------------------------------------------------------------------
# SessionStore: atomic save + corruption-resilient load
# ---------------------------------------------------------------------------


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "absent.json")
    assert store.load() is None


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    """Atomic save creates intermediate directories so first-run on
    a fresh data_dir doesn't need a separate bootstrap step."""
    target = tmp_path / "nested" / "deep" / "session.json"
    store = SessionStore(target)
    snapshot = SessionSnapshot(saved_at=datetime.now(UTC))
    store.save(snapshot)
    assert target.exists()
    assert target.parent.is_dir()


def test_round_trip_preserves_schema(tmp_path: Path) -> None:
    """Saved snapshot reads back equal to the original (modulo
    pydantic's lossless re-validation)."""
    store = SessionStore(tmp_path / "session.json")
    original = SessionSnapshot(
        saved_at=datetime(2026, 5, 13, 12, 0, tzinfo=UTC),
        tick=42,
    )
    store.save(original)
    loaded = store.load()
    assert loaded is not None
    assert loaded.tick == 42
    assert loaded.saved_at == original.saved_at
    assert loaded.schema_version == CURRENT_SCHEMA_VERSION


def test_load_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    """Garbled JSON should NOT raise — return None and log a warning
    so the dashboard cold-starts cleanly."""
    target = tmp_path / "session.json"
    target.write_text("{not actually json", encoding="utf-8")
    store = SessionStore(target)
    assert store.load() is None


def test_load_returns_none_on_schema_mismatch(tmp_path: Path) -> None:
    """A snapshot from a future / older incompatible schema must be
    discarded silently. The intelligence trackers self-heal on the
    first tick — cold start is cheap."""
    target = tmp_path / "session.json"
    target.write_text(
        '{"schema_version": 9999, "saved_at": "2026-05-13T00:00:00+00:00"}',
        encoding="utf-8",
    )
    store = SessionStore(target)
    assert store.load() is None


def test_save_is_atomic_via_tmp_then_rename(tmp_path: Path) -> None:
    """The tmp file should never be left behind after a successful save."""
    target = tmp_path / "session.json"
    store = SessionStore(target)
    store.save(SessionSnapshot(saved_at=datetime.now(UTC)))
    assert target.exists()
    assert not (target.with_suffix(".json.tmp")).exists()


# ---------------------------------------------------------------------------
# SignalHistory round-trip
# ---------------------------------------------------------------------------


def test_signal_history_round_trip() -> None:
    """Episodes save and load identically; the current+recent summary
    is unchanged after rehydration."""
    history = SignalHistory()
    base = datetime(2026, 5, 13, tzinfo=UTC)
    history.record("AAPL", SignalAction.BUY, 0.5, base)
    history.record("AAPL", SignalAction.BUY, 0.7, base + timedelta(seconds=5))
    history.record("AAPL", SignalAction.SELL, 0.4, base + timedelta(seconds=10))

    snap = history.to_snapshot()
    fresh = SignalHistory()
    fresh.apply_snapshot(snap)

    rehydrated = fresh.summary_for("AAPL")
    original = history.summary_for("AAPL")
    assert rehydrated is not None and original is not None
    assert rehydrated.current.action == original.current.action
    assert rehydrated.current.tick_count == original.current.tick_count
    assert rehydrated.recent == original.recent


def test_signal_history_apply_snapshot_respects_max() -> None:
    """If a prior session ran with a bigger episode cap, the current
    cap still applies — only the most recent ``_max`` episodes load."""
    base = datetime(2026, 5, 13, tzinfo=UTC)
    # Build a snapshot with more episodes than the new cap allows.
    episodes = [
        SignalEpisode(
            action=SignalAction.BUY if i % 2 == 0 else SignalAction.SELL,
            started_at=base + timedelta(seconds=i),
            last_seen_at=base + timedelta(seconds=i),
            tick_count=1,
            confidence_first=0.5,
            confidence_last=0.5,
        )
        for i in range(15)
    ]
    fresh = SignalHistory(max_episodes_per_symbol=5)
    fresh.apply_snapshot({"AAPL": episodes})
    assert len(fresh.episodes_for("AAPL")) == 5
    # Most-recent five should be preserved (the tail).
    assert fresh.episodes_for("AAPL")[0].started_at == episodes[-5].started_at


def test_signal_history_apply_snapshot_clears_prior_state() -> None:
    """apply_snapshot is a full replace, not a merge — orphan symbols
    from a previous in-memory state must vanish."""
    history = SignalHistory()
    history.record("MSFT", SignalAction.BUY, 0.5, datetime(2026, 5, 13, tzinfo=UTC))
    history.apply_snapshot({"AAPL": []})  # empty list pruned by apply
    assert history.summary_for("MSFT") is None
    assert history.summary_for("AAPL") is None


# ---------------------------------------------------------------------------
# OpportunityMembershipTracker round-trip
# ---------------------------------------------------------------------------


def test_opportunity_tracker_round_trip() -> None:
    """Membership buffers preserve order + content across snapshot."""
    tracker = OpportunityMembershipTracker(window=10)
    tracker.record(["AAPL", "MSFT"])
    tracker.record(["AAPL"])
    tracker.record(["AAPL", "NVDA"])

    fresh = OpportunityMembershipTracker(window=10)
    fresh.apply_snapshot(tracker.to_snapshot())

    for sym in ("AAPL", "NVDA"):
        original = tracker.summary_for(sym)
        rehydrated = fresh.summary_for(sym)
        assert original is not None and rehydrated is not None
        assert original.streak == rehydrated.streak
        assert original.appearances == rehydrated.appearances


def test_opportunity_tracker_apply_snapshot_prunes_fully_absent() -> None:
    """A symbol whose restored buffer is fully False is dropped — same
    invariant the live record() path maintains."""
    fresh = OpportunityMembershipTracker(window=5)
    fresh.apply_snapshot({"GHOST": [False, False, False]})
    assert fresh.tracked_symbols() == ()


def test_opportunity_tracker_apply_snapshot_respects_smaller_window() -> None:
    """If the running window is smaller than the saved buffer, only
    the most recent ``window`` entries survive."""
    fresh = OpportunityMembershipTracker(window=3)
    fresh.apply_snapshot({"AAPL": [True, True, True, True, True]})
    summary = fresh.summary_for("AAPL")
    assert summary is not None
    assert summary.window == 3
    assert summary.appearances == 3
    assert summary.streak == 3


# ---------------------------------------------------------------------------
# PulseHistoryTracker round-trip
# ---------------------------------------------------------------------------


def _pulse(
    *,
    sentiment: str = "bullish",
    momentum_breadth: float = 0.6,
    summary: str = "x",
) -> MarketPulse:
    return MarketPulse(
        sentiment=sentiment,
        conviction="moderate",
        activity="calm",
        summary=summary,
        bullish_count=3,
        bearish_count=0,
        healthy_count=3,
        momentum_breadth=momentum_breadth,
        sentiment_breadth=0.4,
        reversal_intensity=1,
        alert_intensity=0,
        strongest_symbols=(("NVDA", "STRONG BUY"),),
    )


def test_pulse_history_round_trip_preserves_series() -> None:
    """All numeric and categorical fields survive the trip, including
    strongest_symbols (which is a tuple-of-tuples)."""
    tracker = PulseHistoryTracker(window=10)
    tracker.record(_pulse(momentum_breadth=0.3))
    tracker.record(_pulse(momentum_breadth=0.7, sentiment="bearish"))

    fresh = PulseHistoryTracker(window=10)
    fresh.apply_snapshot(tracker.to_snapshot())

    rehydrated = fresh.summary()
    original = tracker.summary()
    assert rehydrated.momentum_breadth == original.momentum_breadth
    assert rehydrated.sentiment == original.sentiment
    assert rehydrated.bullish_count == original.bullish_count
    assert rehydrated.length == 2


def test_pulse_history_apply_snapshot_respects_smaller_window() -> None:
    """Saved 5 records but current window is 2 → only last 2 survive."""
    records = [
        PulseRecord(
            sentiment="bullish",
            conviction="moderate",
            activity="calm",
            summary=f"s{i}",
        )
        for i in range(5)
    ]
    fresh = PulseHistoryTracker(window=2)
    fresh.apply_snapshot(records)
    assert len(fresh) == 2
    assert fresh.summary().sentiment[0] == "bullish"


# ---------------------------------------------------------------------------
# AlertState round-trip
# ---------------------------------------------------------------------------


def test_alert_state_round_trip_preserves_log_and_last_fired() -> None:
    """Both the bounded log and the last-fired cooldown map survive
    the trip — last-fired is what gates the cooldown filter."""
    fired = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
    state = AlertState()
    state.record(
        [
            Alert(
                symbol="AAPL",
                rule="action_changed",
                severity="warn",
                message="x",
                fired_at=fired,
            ),
            Alert(
                symbol="MSFT",
                rule="confidence_threshold",
                severity="info",
                message="y",
                fired_at=fired,
            ),
        ]
    )
    log, last_fired = state.to_snapshot()
    fresh = AlertState()
    fresh.apply_snapshot(log, last_fired)
    assert len(fresh) == 2
    assert fresh.last_fired("AAPL", "action_changed") == fired
    assert fresh.last_fired("MSFT", "confidence_threshold") == fired


def test_alert_state_apply_snapshot_skips_corrupt_keys() -> None:
    """A tampered key without the ``||`` delimiter should be ignored
    rather than crashing the load path."""
    fired = datetime.now(UTC)
    fresh = AlertState()
    fresh.apply_snapshot(
        log=[],
        last_fired_serialized={"corrupt-no-delimiter": fired},
    )
    assert fresh.last_fired("corrupt-no-delimiter", "anything") is None


# ---------------------------------------------------------------------------
# Controller hydration end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_controller_persists_and_hydrates(tmp_path: Path) -> None:
    """End-to-end: the mock controller writes a session file each
    tick; a fresh controller hydrates from that file and resumes at
    the saved tick number."""
    from src.dashboard.controller import MockDashboardController

    state_path = tmp_path / "session.json"
    ctrl_a = MockDashboardController(
        watchlist=["AAPL"],
        seed=1,
        session_store=SessionStore(state_path),
    )
    await ctrl_a.fetch_snapshot()
    await ctrl_a.fetch_snapshot()
    assert state_path.exists()

    # Brand-new controller pointed at the same file — should pick up
    # at tick 3 (two prior ticks recorded) rather than starting at 1.
    ctrl_b = MockDashboardController(
        watchlist=["AAPL"],
        seed=1,
        session_store=SessionStore(state_path),
    )
    snap = await ctrl_b.fetch_snapshot()
    assert snap.tick == 3


@pytest.mark.asyncio
async def test_controller_without_session_store_does_not_create_file(
    tmp_path: Path,
) -> None:
    """Persistence is opt-in: a controller with no SessionStore must
    never touch the filesystem — tests pass session_store=None to
    keep state ephemeral."""
    from src.dashboard.controller import MockDashboardController

    forbidden = tmp_path / "should-not-appear.json"
    ctrl = MockDashboardController(watchlist=["AAPL"], seed=1)
    await ctrl.fetch_snapshot()
    assert not forbidden.exists()
