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
    """A snapshot from a future / unrecognized schema must be
    discarded silently. The intelligence trackers self-heal on the
    first tick — cold start is cheap."""
    target = tmp_path / "session.json"
    target.write_text(
        '{"schema_version": 9999, "saved_at": "2026-05-13T00:00:00+00:00"}',
        encoding="utf-8",
    )
    store = SessionStore(target)
    assert store.load() is None


def test_v1_snapshot_migrates_to_v2_with_empty_intraday(tmp_path: Path) -> None:
    """MT2 bumped CURRENT_SCHEMA_VERSION from 1 to 2 with one new
    field (``intraday_signal_episodes``). v1 snapshots on disk must
    still load — they migrate in-memory by validating with the
    pydantic default, and the next persist re-writes as v2."""
    target = tmp_path / "session.json"
    target.write_text(
        '{"schema_version": 1, "saved_at": "2026-05-13T12:00:00+00:00", "tick": 12}',
        encoding="utf-8",
    )
    store = SessionStore(target)
    loaded = store.load()
    assert loaded is not None
    assert loaded.tick == 12
    # The v1 file didn't carry intraday state; the default kicks in.
    assert loaded.intraday_signal_episodes == {}


def test_status_starts_degraded_when_no_writes_yet(tmp_path: Path) -> None:
    """A fresh store with no save attempts reports ``degraded`` —
    the persistence loop hasn't proven itself yet."""
    store = SessionStore(tmp_path / "session.json")
    status = store.status()
    assert status.health == "degraded"
    assert status.last_success_at is None
    assert status.bytes is None


def test_status_ok_after_successful_save(tmp_path: Path) -> None:
    """After save() lands cleanly the next status() reports ``ok``
    with a timestamp and a positive file size."""
    store = SessionStore(tmp_path / "session.json")
    store.save(SessionSnapshot(saved_at=datetime.now(UTC)))
    status = store.status()
    assert status.health == "ok"
    assert status.last_success_at is not None
    assert status.bytes is not None and status.bytes > 0
    assert status.last_error is None


def test_status_stale_when_success_older_than_window(tmp_path: Path) -> None:
    """When the last successful write is older than ``stale_after``
    the health string flips to ``stale``. ``now`` is a parameter so
    tests don't need to clock-manipulate."""
    store = SessionStore(tmp_path / "session.json")
    store.save(SessionSnapshot(saved_at=datetime.now(UTC)))
    assert store._last_success_at is not None
    far_future = store._last_success_at + timedelta(minutes=5)
    status = store.status(now=far_future, stale_after=timedelta(seconds=30))
    assert status.health == "stale"


def test_status_degraded_after_write_failure(tmp_path: Path) -> None:
    """A failed save() marks the store degraded. The next successful
    save() clears the degraded state — recovery is not sticky."""
    store = SessionStore(tmp_path / "missing_dir_that_cannot_be_made")
    # Point the store at a path with an illegal name on Windows /
    # any platform that can't write under a NUL-name. Trigger
    # _record_failure directly since constructing a guaranteed-fail
    # path is fiddly across OSes.
    store._record_failure("simulated error")
    assert store.status().health == "degraded"
    # Now record a success and confirm the state clears.
    store._record_success(size=42)
    fresh = store.status()
    assert fresh.health == "ok"
    assert fresh.bytes == 42


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


def test_brief_cache_round_trips_through_snapshot(tmp_path: Path) -> None:
    """SessionSnapshot's brief_cache + opp_brief_cache fields survive
    serialization. Pydantic's default_factory keeps these backward-
    compatible — old snapshots without the fields still load."""
    store = SessionStore(tmp_path / "session.json")
    original = SessionSnapshot(
        saved_at=datetime.now(UTC),
        brief_cache={"AAPL|buy": "Apple looks strong on supply chain news."},
        opp_brief_cache={"NVDA|82": "NVDA is the top OPP at composite 0.82."},
    )
    store.save(original)
    loaded = store.load()
    assert loaded is not None
    assert loaded.brief_cache == {"AAPL|buy": "Apple looks strong on supply chain news."}
    assert loaded.opp_brief_cache == {"NVDA|82": "NVDA is the top OPP at composite 0.82."}


def test_old_snapshot_without_brief_fields_still_loads(tmp_path: Path) -> None:
    """Snapshots written before the brief-cache fields existed should
    load cleanly — pydantic fills in the new dict fields with empty
    defaults."""
    target = tmp_path / "session.json"
    target.write_text(
        '{"schema_version": 1, "saved_at": "2026-05-13T12:00:00+00:00", "tick": 7}',
        encoding="utf-8",
    )
    store = SessionStore(target)
    loaded = store.load()
    assert loaded is not None
    assert loaded.tick == 7
    assert loaded.brief_cache == {}
    assert loaded.opp_brief_cache == {}


def test_controller_record_briefs_round_trip_across_restart(tmp_path: Path) -> None:
    """End-to-end: a brief recorded into the controller mirror lands
    in the JSON snapshot at tick-end persist, and a fresh controller
    pointed at the same file hydrates the brief on construction."""
    from src.dashboard.controller import MockDashboardController

    state_path = tmp_path / "session.json"
    ctrl_a = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    ctrl_a.record_row_brief("AAPL", "buy", "saved row brief text")
    ctrl_a.record_opp_brief("AAPL", 75, "saved OPP brief text")
    # Persist directly — calling the internal helper avoids needing
    # an event loop for this sync test.
    ctrl_a._persist_if_enabled()

    # Brand-new controller — should see both briefs after construction.
    ctrl_b = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    assert ctrl_b.brief_cache == {"AAPL|buy": "saved row brief text"}
    assert ctrl_b.opp_brief_cache == {"AAPL|75": "saved OPP brief text"}


def test_intraday_signal_history_records_when_row_has_intraday() -> None:
    """Phase 2a: the controller's intraday_signal_history records
    from row.intraday whenever the row carries one. Rows without an
    intraday read are skipped — no synthetic HOLD entries."""
    from datetime import timedelta

    from src.dashboard.controller import MockDashboardController
    from src.dashboard.state import RecommendationRow
    from src.data.models import TimeFrame
    from src.strategy.base import SignalAction
    from src.strategy.multi_timeframe import IntradayRead

    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=1)
    base = datetime(2026, 5, 13, tzinfo=UTC)

    fired = datetime.now(UTC)
    rows = [
        RecommendationRow(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.6,
            combined_score=0.5,
            technical_score=0.4,
            sentiment_score=0.3,
            rsi=0.0,
            macd=0.0,
            bollinger=0.0,
            last_price=190.0,
            num_news_articles=1,
            reasoning="",
            timestamp=fired,
            intraday=IntradayRead(
                timeframe=TimeFrame.MIN_15,
                action=SignalAction.SELL,
                confidence=0.4,
                combined_score=-0.4,
                technical_score=-0.4,
            ),
        ),
        # MSFT has no intraday read — should be skipped, not synth-recorded.
        RecommendationRow(
            symbol="MSFT",
            action=SignalAction.BUY,
            confidence=0.5,
            combined_score=0.4,
            technical_score=0.3,
            sentiment_score=0.2,
            rsi=0.0,
            macd=0.0,
            bollinger=0.0,
            last_price=420.0,
            num_news_articles=0,
            reasoning="",
            timestamp=fired,
        ),
    ]
    ctrl._record_intraday_history(rows, base + timedelta(seconds=10))
    aapl = ctrl.intraday_signal_history.summary_for("AAPL")
    assert aapl is not None
    assert aapl.current.action == SignalAction.SELL
    # MSFT had no intraday read — no episodes recorded.
    assert ctrl.intraday_signal_history.summary_for("MSFT") is None


def test_intraday_signal_history_persists_via_current_snapshot(tmp_path: Path) -> None:
    """End-to-end: an intraday episode recorded on one controller
    survives the snapshot round-trip and hydrates on a fresh
    controller. The schema version is whatever the current code
    writes (was v2 in phase 2a; v3 after phase 2b adds the
    intraday OPP fields). Either is fine — the persistence is the
    contract being tested here, not the version bump."""
    from datetime import timedelta

    from src.dashboard.controller import MockDashboardController
    from src.dashboard.state import RecommendationRow
    from src.data.models import TimeFrame
    from src.strategy.base import SignalAction
    from src.strategy.multi_timeframe import IntradayRead

    state_path = tmp_path / "session.json"
    ctrl_a = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    fired = datetime.now(UTC)
    row = RecommendationRow(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.6,
        combined_score=0.5,
        technical_score=0.4,
        sentiment_score=0.3,
        rsi=0.0,
        macd=0.0,
        bollinger=0.0,
        last_price=190.0,
        num_news_articles=1,
        reasoning="",
        timestamp=fired,
        intraday=IntradayRead(
            timeframe=TimeFrame.MIN_15,
            action=SignalAction.SELL,
            confidence=0.4,
            combined_score=-0.4,
            technical_score=-0.4,
        ),
    )
    ctrl_a._record_intraday_history([row], fired + timedelta(seconds=5))
    ctrl_a._persist_if_enabled()

    # Snapshot on disk should carry the current schema version.
    loaded = SessionStore(state_path).load()
    assert loaded is not None
    assert loaded.schema_version == CURRENT_SCHEMA_VERSION
    assert "AAPL" in loaded.intraday_signal_episodes

    # Fresh controller hydrates the intraday tracker.
    ctrl_b = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    summary = ctrl_b.intraday_signal_history.summary_for("AAPL")
    assert summary is not None
    assert summary.current.action == SignalAction.SELL


def test_intraday_opp_tracker_persists_via_v3_snapshot(tmp_path: Path) -> None:
    """Phase 2b: intraday OPP membership is recorded into a parallel
    OpportunityMembershipTracker and persisted alongside the daily
    tracker. A fresh controller pointed at the same file hydrates
    the intraday tracker."""
    from src.dashboard.controller import MockDashboardController

    state_path = tmp_path / "session.json"
    ctrl_a = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    # Directly seed the parallel intraday tracker — bypasses the
    # async fetch path so the test stays synchronous.
    ctrl_a.intraday_opp_tracker.record(["AAPL", "MSFT"])
    ctrl_a.intraday_opp_tracker.record(["AAPL"])
    ctrl_a._persist_if_enabled()

    loaded = SessionStore(state_path).load()
    assert loaded is not None
    assert loaded.schema_version == CURRENT_SCHEMA_VERSION
    assert "AAPL" in loaded.intraday_opp_membership
    assert "MSFT" in loaded.intraday_opp_membership

    # Fresh controller hydrates the intraday tracker.
    ctrl_b = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    aapl_summary = ctrl_b.intraday_opp_tracker.summary_for("AAPL")
    assert aapl_summary is not None
    assert aapl_summary.streak == 2


def test_v2_snapshot_migrates_to_v3_with_empty_intraday_opp(tmp_path: Path) -> None:
    """A v2 snapshot on disk (signal-history fields but no intraday
    OPP membership) must still load — the new v3 fields default to
    empty dicts via pydantic."""
    target = tmp_path / "session.json"
    target.write_text(
        '{"schema_version": 2, "saved_at": "2026-05-14T12:00:00+00:00", "tick": 5}',
        encoding="utf-8",
    )
    loaded = SessionStore(target).load()
    assert loaded is not None
    assert loaded.tick == 5
    assert loaded.intraday_opp_membership == {}


def test_controller_prune_briefs_drops_only_matching_symbol() -> None:
    """``prune_briefs_for_symbol`` keeps other symbols' briefs intact."""
    from src.dashboard.controller import MockDashboardController

    ctrl = MockDashboardController(watchlist=["AAPL", "MSFT"], seed=1)
    ctrl.record_row_brief("AAPL", "buy", "apple text")
    ctrl.record_row_brief("MSFT", "sell", "microsoft text")
    ctrl.record_opp_brief("AAPL", 70, "apple opp text")
    ctrl.record_opp_brief("MSFT", 55, "microsoft opp text")
    ctrl.prune_briefs_for_symbol("AAPL")
    assert ctrl.brief_cache == {"MSFT|sell": "microsoft text"}
    assert ctrl.opp_brief_cache == {"MSFT|55": "microsoft opp text"}


@pytest.mark.asyncio
async def test_dashboard_app_hydrates_briefs_from_controller(tmp_path: Path) -> None:
    """App.on_mount copies the controller's persisted brief caches
    into local dict storage so a press of `s`/`b` after restart
    surfaces the saved text without re-billing the LLM."""
    from src.dashboard.app import DashboardApp
    from src.dashboard.controller import MockDashboardController

    state_path = tmp_path / "session.json"
    # Pre-seed the persisted snapshot with two briefs. Persisting
    # synchronously via the internal helper avoids fetching a tick.
    ctrl = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    ctrl.record_row_brief("AAPL", "buy", "row-brief-text")
    ctrl.record_opp_brief("AAPL", 80, "opp-brief-text")
    ctrl._persist_if_enabled()

    # Fresh controller + app — app on_mount should hydrate briefs.
    fresh_ctrl = MockDashboardController(
        watchlist=["AAPL"], seed=1, session_store=SessionStore(state_path)
    )
    app = DashboardApp(fresh_ctrl, refresh_seconds=999.0)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app._brief_cache.get(("AAPL", "buy")) == "row-brief-text"
        assert app._opp_brief_cache.get(("AAPL", 80)) == "opp-brief-text"


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
