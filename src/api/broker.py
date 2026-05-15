"""Single-process snapshot tick loop with fan-out to many subscribers.

The :class:`SnapshotBroker` owns *the* tick loop inside an
``esther serve`` process. Every connected SSE client subscribes; each
gets its own size-1, newest-wins :class:`asyncio.Queue` that the
broker writes to as soon as a new :class:`DashboardSnapshot` is
produced.

Design constraints (from the streaming-endpoint requirements):

- **One tick loop per process.** Not one per client. ``fetch_snapshot``
  is expensive (Alpaca + FinBERT) — N parallel pipelines for N clients
  would multiply API cost and produce N slightly-different views of
  the same market.
- **Newest-wins queues.** A slow client that falls behind by N ticks
  never accumulates a stale backlog — they always see the most recent
  snapshot on the next ``queue.get()``. Missing the intermediate ticks
  is harmless: each snapshot is full state.
- **Error-tolerant loop.** A bad Alpaca response can't kill the
  stream. Failures are logged and the loop continues at the next
  interval.
- **Subscribe seeds with cached latest.** New clients (and reconnecting
  ones) see current state immediately — no waiting up to one full
  interval before anything appears on screen.

The broker is intentionally plain (no FastAPI / Starlette imports) so
it stays unit-testable with bare ``asyncio`` and so any future surface
(WebSocket, gRPC, raw TCP) can subscribe via the same interface.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.dashboard.controller import BaseController
    from src.dashboard.state import DashboardSnapshot


_log = get_logger("api.broker")


class SnapshotBroker:
    """Drives a single tick loop and fans snapshots out to subscribers."""

    def __init__(self, controller: BaseController, interval: float) -> None:
        """
        Args:
            controller: The shared snapshot-producing controller. The
                broker is the only thing that should call its
                ``fetch_snapshot`` method while the broker is running.
            interval: Seconds to wait *between* tick completions. With
                a 5s interval and a 3s fetch, the next tick starts ~8s
                after the previous one finished. We deliberately do
                not chase a wall-clock cadence — keeps the loop
                resilient under load instead of stacking overlapping
                fetches.
        """
        if interval <= 0:
            raise ValueError(f"broker interval must be > 0, got {interval}")
        self._controller = controller
        self._interval = interval
        self._latest: DashboardSnapshot | None = None
        self._subscribers: set[asyncio.Queue[DashboardSnapshot]] = set()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def latest(self) -> DashboardSnapshot | None:
        """Most recent snapshot, or None before the first successful tick.

        Reading is safe from any coroutine — the field is atomic-replace
        and never partially updated.
        """
        return self._latest

    @property
    def subscriber_count(self) -> int:
        """For diagnostics / lifecycle assertions."""
        return len(self._subscribers)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Spawn the background tick task.

        Idempotent: calling twice without an intervening ``stop`` is a
        no-op. (We tolerate this because lifespan handlers can fire on
        reload / reconfiguration paths in some servers.)
        """
        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="snapshot-broker")

    async def stop(self) -> None:
        """Signal the loop to exit and wait for it.

        Cancels any in-flight ``fetch_snapshot`` indirectly by setting
        the stop event between awaits — the loop checks before each
        sleep. Returns when the task is gone.
        """
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def subscribe(self) -> asyncio.Queue[DashboardSnapshot]:
        """Register a new subscriber and return its queue.

        The queue has ``maxsize=1`` with newest-wins overwrite: a slow
        consumer cannot accumulate a backlog of stale snapshots — the
        next ``get()`` always yields the latest. Each subscriber owns
        its own queue; broker fan-out is one write per subscriber per
        tick.

        If a snapshot has been seen already, it is enqueued
        immediately so the new subscriber doesn't wait up to one full
        interval before its first ``get()`` returns.
        """
        queue: asyncio.Queue[DashboardSnapshot] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        if self._latest is not None:
            self._put_newest_wins(queue, self._latest)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DashboardSnapshot]) -> None:
        """Drop a subscriber. Safe to call multiple times."""
        self._subscribers.discard(queue)

    async def _loop(self) -> None:
        """The tick loop. Fetches one snapshot, fans out, sleeps, repeats.

        Exceptions in ``fetch_snapshot`` are caught and logged. We do
        *not* propagate them — a transient Alpaca / FinBERT failure
        should not tear down every connected client.
        """
        while not self._stop.is_set():
            try:
                snap = await self._controller.fetch_snapshot()
            except Exception as exc:
                _log.warning("broker.tick_failed", error=str(exc))
            else:
                self._latest = snap
                self._fanout(snap)
            # Interval-style scheduling: sleep AFTER the fetch, not at
            # a fixed wall-clock cadence. wait_for races the stop event
            # against the sleep so stop() returns quickly.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)

    def _fanout(self, snap: DashboardSnapshot) -> None:
        """Write ``snap`` to every subscriber's queue, newest-wins.

        Iterates a snapshot of ``_subscribers`` so a concurrent
        unsubscribe during fan-out doesn't mutate-during-iteration.
        """
        for queue in list(self._subscribers):
            self._put_newest_wins(queue, snap)

    @staticmethod
    def _put_newest_wins(
        queue: asyncio.Queue[DashboardSnapshot],
        snap: DashboardSnapshot,
    ) -> None:
        """Put ``snap`` into ``queue``, dropping any stale entry first.

        With ``maxsize=1`` the queue is either empty or holds one
        already-queued snapshot. If full, we drain it (the slow
        consumer hasn't read yet) and write the fresher snapshot. The
        consumer's next ``get()`` then returns the latest data.
        """
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(snap)
