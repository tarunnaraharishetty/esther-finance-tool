"""Unit tests for :class:`SnapshotBroker`.

The broker is the load-bearing piece of the streaming endpoint: one
tick loop per process, fanned out to many subscribers. These tests
pin down the contract every consumer (SSE handler, hypothetical
WebSocket handler) depends on.

We use a tiny in-test fake controller instead of the mock
``MockDashboardController`` from ``src.dashboard.controller`` — the
fake here lets each test inject failures, custom snapshots, and slow
fetches without dragging in the synthetic-OHLCV machinery.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from src.api.broker import SnapshotBroker


@dataclass
class _Snap:
    """Tiny snapshot stand-in. The broker is generic over snapshot
    shape — it only stores and forwards. Using a small dataclass keeps
    the tests focused on the broker's behavior rather than the rich
    real DashboardSnapshot.
    """

    tick: int


class _FakeController:
    """Programmable fake controller for broker tests.

    - ``fetch_snapshot`` returns a fresh ``_Snap`` with a monotonically
      increasing tick.
    - If ``raise_next`` is set, the next call raises that exception
      instead, simulating a transient Alpaca failure.
    - ``fetch_count`` tracks how many times ``fetch_snapshot`` was
      called so tests can assert exact call counts.
    """

    def __init__(self) -> None:
        self.fetch_count = 0
        self.raise_next: BaseException | None = None
        self.delay: float = 0.0

    async def fetch_snapshot(self) -> _Snap:
        self.fetch_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_next is not None:
            err = self.raise_next
            self.raise_next = None
            raise err
        return _Snap(tick=self.fetch_count)


def _broker(controller: Any, *, interval: float = 0.02) -> SnapshotBroker:
    """Build a broker with a short interval suitable for fast tests."""
    return SnapshotBroker(controller, interval=interval)


async def _wait_for(predicate: Any, deadline_seconds: float = 1.0, step: float = 0.005) -> None:
    """Poll ``predicate`` until truthy or the deadline elapses.

    Avoids fragile ``asyncio.sleep(0.5)`` patterns by waking up
    frequently and exiting as soon as the condition is met. The
    parameter is named ``deadline_seconds`` rather than ``timeout``
    because the ASYNC109 lint rule reserves the latter for
    ``asyncio.timeout`` context managers.
    """
    deadline = asyncio.get_event_loop().time() + deadline_seconds
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(step)
    raise AssertionError(f"timed out waiting for predicate after {deadline_seconds}s")


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_broker_rejects_non_positive_interval() -> None:
    """A zero or negative interval is a programming error — fail fast
    at construction rather than spinning a hot loop."""
    with pytest.raises(ValueError, match="interval must be > 0"):
        SnapshotBroker(_FakeController(), interval=0.0)
    with pytest.raises(ValueError, match="interval must be > 0"):
        SnapshotBroker(_FakeController(), interval=-1.0)


async def test_start_stop_is_clean() -> None:
    """Start → stop completes without leaving the task hanging."""
    broker = _broker(_FakeController())
    assert not broker.is_running
    await broker.start()
    assert broker.is_running
    await broker.stop()
    assert not broker.is_running


async def test_double_start_is_idempotent() -> None:
    """Calling start twice doesn't spawn two tasks (which would
    double the fetch rate against the same controller)."""
    ctrl = _FakeController()
    broker = _broker(ctrl, interval=0.01)
    await broker.start()
    first_task = broker._task
    await broker.start()
    assert broker._task is first_task
    await broker.stop()


# --------------------------------------------------------------------------- #
# Fan-out
# --------------------------------------------------------------------------- #


async def test_first_tick_populates_latest() -> None:
    """After one tick, broker.latest holds the most recent snapshot."""
    ctrl = _FakeController()
    broker = _broker(ctrl)
    await broker.start()
    try:
        await _wait_for(lambda: broker.latest is not None)
        assert broker.latest is not None
        assert broker.latest.tick >= 1
    finally:
        await broker.stop()


async def test_subscriber_receives_each_tick() -> None:
    """A subscriber's queue gets fresh snapshots as ticks occur."""
    ctrl = _FakeController()
    broker = _broker(ctrl, interval=0.01)
    await broker.start()
    queue = broker.subscribe()
    try:
        ticks_seen: list[int] = []
        for _ in range(3):
            snap = await asyncio.wait_for(queue.get(), timeout=1.0)
            ticks_seen.append(snap.tick)
        # Ticks may not be strictly consecutive if the broker
        # outpaces queue.get(), but they must be monotonically
        # increasing (newest-wins semantics).
        assert ticks_seen == sorted(ticks_seen)
        assert ticks_seen[-1] > ticks_seen[0]
    finally:
        broker.unsubscribe(queue)
        await broker.stop()


async def test_multiple_subscribers_get_the_same_snapshot() -> None:
    """Fan-out: every subscriber sees each tick from the single loop.

    This is the property that prevents N clients from causing N
    pipelines worth of work — one fetch, N writes.
    """
    ctrl = _FakeController()
    broker = _broker(ctrl, interval=0.05)
    await broker.start()
    q1 = broker.subscribe()
    q2 = broker.subscribe()
    q3 = broker.subscribe()
    try:
        # Wait for at least one tick to arrive on q1; then assert q2/q3
        # have the same value queued (or about to be).
        s1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        s2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        s3 = await asyncio.wait_for(q3.get(), timeout=1.0)
        # All three see ticks within a small window — same broker tick
        # number, modulo the broker producing a fresher one mid-test.
        assert {s1.tick, s2.tick, s3.tick} <= {s1.tick, s1.tick + 1}, (
            f"subscribers diverged: {s1.tick} {s2.tick} {s3.tick}"
        )
    finally:
        broker.unsubscribe(q1)
        broker.unsubscribe(q2)
        broker.unsubscribe(q3)
        await broker.stop()


# --------------------------------------------------------------------------- #
# Subscribe-flushes-latest
# --------------------------------------------------------------------------- #


async def test_subscribe_seeds_with_cached_latest() -> None:
    """A subscriber connecting AFTER the first tick gets the cached
    snapshot immediately — no waiting up to one full interval."""
    ctrl = _FakeController()
    broker = _broker(ctrl, interval=10.0)  # ~never tick during the test
    await broker.start()
    try:
        await _wait_for(lambda: broker.latest is not None)
        # By now the broker has its first snapshot cached. New
        # subscribers should see it on their next get() without
        # waiting for the second tick (which is 10s away).
        late = broker.subscribe()
        snap = await asyncio.wait_for(late.get(), timeout=0.5)
        assert snap.tick == broker.latest.tick  # type: ignore[union-attr]
    finally:
        await broker.stop()


async def test_subscribe_before_first_tick_does_not_seed() -> None:
    """If no snapshot exists yet, the subscriber's queue starts empty —
    not blocked, not pre-loaded with stale data."""
    ctrl = _FakeController()
    ctrl.delay = 1.0  # delay first fetch so we can sneak a subscribe in
    broker = _broker(ctrl, interval=0.5)
    await broker.start()
    try:
        queue = broker.subscribe()
        # Should be empty right now — broker is mid-fetch, no latest.
        assert queue.empty()
    finally:
        await broker.stop()


# --------------------------------------------------------------------------- #
# Newest-wins overflow
# --------------------------------------------------------------------------- #


async def test_slow_consumer_only_sees_newest_snapshot() -> None:
    """If a subscriber doesn't drain its queue, the broker overwrites
    stale entries with the latest. The next get() yields the freshest
    tick, not the first one that piled up."""
    ctrl = _FakeController()
    broker = _broker(ctrl, interval=0.01)
    await broker.start()
    queue = broker.subscribe()
    try:
        # Let several ticks fan out without reading.
        await _wait_for(lambda: ctrl.fetch_count >= 5, deadline_seconds=1.0)
        snap = await asyncio.wait_for(queue.get(), timeout=0.5)
        # We should get a recent tick (≥ 5 by construction), not tick 1.
        assert snap.tick >= 4
    finally:
        broker.unsubscribe(queue)
        await broker.stop()


# --------------------------------------------------------------------------- #
# Error tolerance
# --------------------------------------------------------------------------- #


async def test_fetch_error_does_not_kill_loop() -> None:
    """A transient fetch failure must be caught — one bad tick can't
    take down every connected SSE client."""
    ctrl = _FakeController()
    ctrl.raise_next = RuntimeError("alpaca rate limit, retry")
    broker = _broker(ctrl, interval=0.01)
    await broker.start()
    try:
        # Even though the first fetch raises, subsequent ticks succeed
        # and a subscriber eventually gets one.
        queue = broker.subscribe()
        snap = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert snap.tick >= 1
        broker.unsubscribe(queue)
    finally:
        await broker.stop()


# --------------------------------------------------------------------------- #
# Unsubscribe
# --------------------------------------------------------------------------- #


async def test_unsubscribe_removes_queue_from_fanout() -> None:
    """After unsubscribe, no further ticks land in that queue."""
    ctrl = _FakeController()
    broker = _broker(ctrl, interval=0.02)
    await broker.start()
    queue = broker.subscribe()
    try:
        await asyncio.wait_for(queue.get(), timeout=1.0)
        broker.unsubscribe(queue)
        # Drain anything currently in the queue, then assert that no
        # more snapshots arrive over the next few ticks.
        while not queue.empty():
            queue.get_nowait()
        baseline_fetches = ctrl.fetch_count
        await asyncio.sleep(0.1)  # ~5 ticks at 20ms
        assert ctrl.fetch_count > baseline_fetches  # broker still ticking
        assert queue.empty()
    finally:
        await broker.stop()


async def test_unsubscribe_is_safe_to_call_twice() -> None:
    """Defensive: double-unsubscribe must be a no-op rather than a
    KeyError, since cleanup paths can race."""
    broker = _broker(_FakeController())
    await broker.start()
    queue = broker.subscribe()
    broker.unsubscribe(queue)
    broker.unsubscribe(queue)  # must not raise
    assert broker.subscriber_count == 0
    await broker.stop()
