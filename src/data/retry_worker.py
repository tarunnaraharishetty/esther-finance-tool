"""Async drain loop for :class:`RetryQueue`.

The worker reads due entries, attempts the configured retry function
for each, and updates queue state based on the outcome. It does
*not* know about the fundamentals chain directly — it takes a
``retry_fn`` callable so the queue + worker can be exercised against
any retryable workload in tests (and so the dependency layering stays
clean: ``src/data/`` has no awareness of ``intelligence/fundamentals``).

Lifecycle
---------
:meth:`start` schedules an asyncio task; :meth:`stop` cancels it.
The task loops forever calling :meth:`tick` at ``interval_seconds``.
A single bad entry can't kill the loop — per-entry exceptions are
caught and logged.

When to use it
--------------
``RetryWorker`` is opt-in via
:attr:`Settings.retry_queue_worker_enabled` — production deployments
turn it on to recover from transient provider failures without user
re-clicks. Test environments and the local TUI keep it off so
background API calls don't surprise the developer.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from src.data.retry_queue import RetryQueue
from src.utils.logging import get_logger

log = get_logger(__name__)


# Sentinel value the worker reads from the retry function to detect
# "transient failure — please reschedule". Anything else (success or
# permanent failure) the worker derives from the type of result or
# the raised exception.
class TransientRetryFailure(Exception):
    """Raised by the retry function to signal "please reschedule".

    The ``errors`` tuple becomes the entry's ``last_errors`` on the
    next reschedule. Distinct from a generic exception so a stray
    KeyError from a bug doesn't masquerade as a retryable signal.

    ``retry_after_seconds`` is a server-supplied cooldown hint (e.g.
    the ``Retry-After`` header on a 429). When provided and larger
    than the worker's default backoff, the queue uses it as the
    reschedule floor so we don't keep hammering an upstream that
    explicitly asked us to wait.
    """

    def __init__(
        self,
        errors: tuple[str, ...],
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__("; ".join(errors) or "transient retry failure")
        self.errors = errors
        self.retry_after_seconds = retry_after_seconds


class PermanentRetryFailure(Exception):
    """Raised to immediately mark an entry permanently failed.

    Used when the retry function determines the failure won't recover
    on retry (e.g., the API key was rotated out, the symbol was
    delisted). Bypasses the attempt counter.
    """

    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("; ".join(errors) or "permanent retry failure")
        self.errors = errors


RetryFn = Callable[[str], Awaitable[Any]]


class RetryWorker:
    """Background asyncio loop draining a :class:`RetryQueue`.

    Args:
        queue: The queue to drain.
        retry_fn: Async callable invoked once per due symbol. Return
            normally on success; raise :class:`TransientRetryFailure`
            to reschedule with backoff; raise
            :class:`PermanentRetryFailure` to mark the entry failed
            immediately. Any other exception is treated as transient
            (defensive — a bug in the retry function shouldn't strand
            the queue, but operators see the log line).
        interval_seconds: Loop tick cadence. Tighter intervals
            improve responsiveness; loose intervals reduce idle wakeups.
    """

    def __init__(
        self,
        queue: RetryQueue,
        retry_fn: RetryFn,
        *,
        interval_seconds: float = 60.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {interval_seconds!r}"
            )
        self._queue = queue
        self._retry_fn = retry_fn
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Schedule the loop task. Safe to call once.

        Re-starting an already-running worker is a no-op so harness
        code (e.g., a FastAPI lifespan handler that fires twice in
        an edge case) doesn't spawn duplicate loops.
        """
        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="retry-worker")

    async def stop(self) -> None:
        """Cancel the loop. Safe to call when not running."""
        if not self.is_running:
            return
        self._stop.set()
        task = self._task
        if task is not None:
            task.cancel()
            # Cancellation is the normal exit path. Any other
            # exception was already logged inside _run.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._task = None

    async def tick(self) -> int:
        """Run a single drain pass. Returns the number of entries processed.

        Exposed for tests so the loop body can be exercised without
        an asyncio scheduler. Also used internally by :meth:`_run`.
        """
        due = self._queue.due_entries()
        processed = 0
        for entry in due:
            await self._process(entry.symbol)
            processed += 1
        return processed

    async def _run(self) -> None:
        log.info("retry_worker.start", interval_seconds=self._interval)
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except Exception as exc:
                    # A bug inside the loop body must not kill the
                    # worker. Log and continue — the next tick may
                    # succeed.
                    log.warning("retry_worker.tick_failed", error=str(exc))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                except TimeoutError:
                    continue
        finally:
            log.info("retry_worker.stop")

    async def _process(self, symbol: str) -> None:
        try:
            await self._retry_fn(symbol)
        except TransientRetryFailure as exc:
            self._queue.record_failure(
                symbol,
                exc.errors,
                retry_after_seconds=exc.retry_after_seconds,
            )
            return
        except PermanentRetryFailure as exc:
            # Force the entry past max_attempts so it lands at
            # permanently_failed in one shot. We do this by feeding
            # the queue's recorder repeatedly until the status flips.
            # Defensive: cap the loop so a misbehaving queue can't
            # spin.
            for _ in range(20):
                updated = self._queue.record_failure(symbol, exc.errors)
                if updated is None:
                    return
                from src.data.retry_queue import RetryStatus

                if updated.status is RetryStatus.PERMANENTLY_FAILED:
                    return
            return
        except Exception as exc:
            # Unexpected exception. Treat as transient — don't burn
            # all retries on a single Python bug — and surface in the
            # log so the operator can investigate.
            log.warning(
                "retry_worker.unexpected_error",
                symbol=symbol,
                error=str(exc),
            )
            self._queue.record_failure(symbol, (f"unexpected: {exc}",))
            return
        # No exception → success.
        self._queue.record_success(symbol)


__all__ = [
    "PermanentRetryFailure",
    "RetryFn",
    "RetryWorker",
    "TransientRetryFailure",
]
