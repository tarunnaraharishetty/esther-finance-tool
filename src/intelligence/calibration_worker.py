"""Calibration recorder + maturation worker.

Closes the data-flow loop for the calibration system:

* :class:`CalibrationRecorder` — given a snapshot of
  :class:`~src.intelligence.analyzer.technical.TechnicalScores` and
  the symbol's current price, write one
  :class:`~src.intelligence.calibration.Observation` per configured
  ``(score, outcome)`` pairing. Hook the analyzer endpoint / snapshot
  loop to this so live data accumulates into the store.

* :class:`CalibrationMaturationWorker` — background async loop that
  finds observations whose horizon has elapsed, looks up the realized
  price at maturation, and settles them via
  :meth:`CalibrationStore.record_outcome`.

Worker lifecycle and isolation
------------------------------
Same shape as :class:`~src.data.retry_worker.RetryWorker`. ``start``
schedules an asyncio task; ``stop`` cancels it. A bad lookup never
kills the loop — per-observation exceptions are caught and logged,
the next observation is processed.

Why ``derive_outcome`` is a pure function
-----------------------------------------
The settle decision must be auditable without state. Given
``(starting_price, ending_price, outcome_name)`` it returns the
binary hit. Tests pin the truth table. Pre-existing observations
without ``starting_price`` (schema-v1 rows) can't be settled — the
worker explicitly skips them and surfaces the count in logs.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from src.data import cache as bar_cache
from src.data.models import TimeFrame
from src.intelligence.analyzer.calibration import PAIRINGS
from src.intelligence.analyzer.technical import TechnicalScores
from src.intelligence.calibration import CalibrationStore, Observation
from src.utils.logging import get_logger

log = get_logger(__name__)


# Async callable that returns the symbol's price at-or-near ``at``.
# Returns ``None`` when no price is available (data gap, weekend,
# delisted symbol). Production wires this to the bars cache + the
# market_data service; tests inject a fake mapping.
PriceLookup = Callable[[str, datetime], Awaitable[float | None]]


# Outcome derivation. Pre-defined so the worker doesn't have to dispatch
# on a string at settle time. Returning ``None`` means "can't grade" —
# the worker skips the observation rather than recording a false outcome.
def derive_outcome(
    starting_price: float | None,
    ending_price: float | None,
    outcome_name: str,
) -> bool | None:
    """Grade a binary outcome from a price pair.

    Returns ``None`` when grading is impossible: missing/non-positive
    start, missing/non-finite end, unknown outcome name. The worker
    skips ``None`` rather than fabricating a settlement.

    A delta of exactly zero grades as ``False`` for both directional
    outcomes — the symbol moved nothing, neither hit fires. Documented
    and pinned in tests; in practice daily-bar zero deltas are vanishing-
    rare.
    """
    if (
        starting_price is None
        or ending_price is None
        or not math.isfinite(starting_price)
        or not math.isfinite(ending_price)
        or starting_price <= 0
    ):
        return None
    delta = (ending_price - starting_price) / starting_price
    if outcome_name == "return_positive":
        return delta > 0
    if outcome_name == "return_negative":
        return delta < 0
    return None


@dataclass(frozen=True)
class CalibrationRecorder:
    """Write one observation per applicable pairing on demand.

    Doesn't mutate state itself — delegates all writes to
    :class:`CalibrationStore.record_observation`. Returned ids are the
    handles the maturation worker eventually settles.
    """

    store: CalibrationStore
    horizon_days: int
    pairings: tuple[tuple[str, str], ...] = PAIRINGS

    def record_scores(
        self,
        *,
        symbol: str,
        technicals: TechnicalScores,
        observed_at: datetime,
        starting_price: float,
    ) -> list[int]:
        """Record one observation per pairing whose score is computable.

        A pairing whose ``score_name`` is ``None`` on the
        ``TechnicalScores`` is skipped silently. Returns the list of
        observation ids written (so a caller that wants to settle
        synchronously in a backfill flow has the handles).
        """
        ids: list[int] = []
        for score_name, outcome_name in self.pairings:
            value = getattr(technicals, score_name, None)
            if value is None:
                continue
            oid = self.store.record_observation(
                symbol=symbol,
                score_name=score_name,
                score_value=float(value),
                observed_at=observed_at,
                horizon_days=self.horizon_days,
                outcome_name=outcome_name,
                starting_price=float(starting_price),
            )
            ids.append(oid)
        return ids


class CalibrationMaturationWorker:
    """Background loop draining :meth:`CalibrationStore.matured_unsettled`.

    Lifecycle and exception isolation mirror
    :class:`~src.data.retry_worker.RetryWorker`. ``start`` schedules
    an asyncio task; ``stop`` cancels. A bad price lookup never
    propagates — the worker logs and moves on.

    Args:
        store: The :class:`CalibrationStore` to drain.
        price_lookup: Async callable returning the symbol's price
            at (or just past) the given moment, or ``None`` when no
            price is available.
        interval_seconds: Loop cadence. Defaults to 1 hour because
            maturation horizons are typically days and there's no
            reason to drain more aggressively than that.
    """

    def __init__(
        self,
        store: CalibrationStore,
        price_lookup: PriceLookup,
        *,
        interval_seconds: float = 3600.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {interval_seconds!r}"
            )
        self._store = store
        self._lookup = price_lookup
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Schedule the loop task. Idempotent on duplicate calls."""
        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(
            self._run(), name="calibration-maturation-worker"
        )

    async def stop(self) -> None:
        """Cancel the loop. Safe when not running."""
        if not self.is_running:
            return
        self._stop.set()
        task = self._task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._task = None

    async def tick(self) -> int:
        """Run a single drain pass. Returns the number of observations settled.

        Exposed publicly so tests can exercise the loop body without
        an asyncio scheduler, and so an operator can trigger a one-
        shot maturation from a shell session.
        """
        matured = self._store.matured_unsettled()
        settled = 0
        skipped_no_price = 0
        skipped_no_start = 0
        skipped_ungradable = 0
        for obs in matured:
            outcome = await self._settle_one(obs)
            if outcome is None:
                continue
            kind, value = outcome
            if kind == "settled":
                settled += 1
            elif kind == "no_price":
                skipped_no_price += 1
            elif kind == "no_start":
                skipped_no_start += 1
            elif kind == "ungradable":
                skipped_ungradable += 1
            del value  # only used for the type bound
        log.info(
            "calibration.maturation.tick",
            matured=len(matured),
            settled=settled,
            skipped_no_price=skipped_no_price,
            skipped_no_start=skipped_no_start,
            skipped_ungradable=skipped_ungradable,
        )
        return settled

    async def _settle_one(
        self, obs: Observation
    ) -> tuple[str, None] | None:
        """Resolve a single observation; return the disposition tag.

        Disposition tags drive the per-tick log counters:
            * ``"settled"``   — outcome derived and persisted.
            * ``"no_start"``  — schema-v1 row with no starting_price.
            * ``"no_price"``  — price lookup returned None or raised.
            * ``"ungradable"`` — derive_outcome refused (unknown
              outcome name, non-finite end price, etc.).
        Returning ``None`` indicates a bug in the loop body itself,
        not an expected skip.
        """
        if obs.starting_price is None:
            return ("no_start", None)
        try:
            ending_price = await self._lookup(obs.symbol, obs.matures_at)
        except Exception as exc:
            log.warning(
                "calibration.maturation.lookup_failed",
                observation_id=obs.id,
                symbol=obs.symbol,
                error=str(exc),
            )
            return ("no_price", None)
        if ending_price is None:
            return ("no_price", None)
        outcome = derive_outcome(obs.starting_price, ending_price, obs.outcome_name)
        if outcome is None:
            return ("ungradable", None)
        self._store.record_outcome(obs.id, outcome)
        return ("settled", None)

    async def _run(self) -> None:
        log.info("calibration_worker.start", interval_seconds=self._interval)
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except Exception as exc:
                    # Loop-body bugs must not stop the worker. The
                    # next tick may succeed once whatever caused the
                    # exception clears.
                    log.warning(
                        "calibration_worker.tick_failed", error=str(exc)
                    )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._interval
                    )
                except TimeoutError:
                    continue
        finally:
            log.info("calibration_worker.stop")


async def bars_cache_price_lookup(symbol: str, at: datetime) -> float | None:
    """Resolve a daily-bar close at or just after ``at`` from the bars cache.

    Production ``PriceLookup`` for :class:`CalibrationMaturationWorker`.
    Reads the same on-disk cache the snapshot loop writes to, so a
    watchlist symbol whose bars have been ticking will always have a
    bar to grade against once ``at`` is reached.

    Returns ``None`` (never raises) when:

    * No cache file exists for the symbol — typically because the
      symbol rotated off the watchlist before the bar at ``at`` was
      fetched. The worker logs as ``no_price`` and tries again next
      tick.
    * The cache exists but contains no rows on or after ``at`` —
      bars haven't extended forward to maturity yet.
    * Any I/O / parse exception on the cache file itself. Calibration
      is observability; a corrupt cache file mustn't break the worker.

    We use the first row at or after ``at`` rather than the closest
    row. The bar that covers (or comes immediately after) the
    maturation moment is the right one — we don't want to look
    backwards in time, because a bar earlier than ``at`` doesn't yet
    reflect what happened during the horizon window.
    """
    try:
        df = bar_cache.read_bars(symbol, TimeFrame.DAY_1)
    except Exception as exc:
        log.warning(
            "calibration.price_lookup.cache_read_failed",
            symbol=symbol,
            error=str(exc),
        )
        return None
    if df is None or df.empty:
        return None
    # Daily bars are indexed at the *start* of the trading day (typically
    # midnight UTC). A maturation ``at`` of 2026-05-17 06:00 UTC should
    # match the bar at 2026-05-17 00:00 UTC — the close of that
    # calendar day reflects the horizon's outcome. Floor ``at`` to the
    # day boundary before the comparison so intraday timestamps don't
    # accidentally skip the bar that covers them.
    at_day = at.replace(hour=0, minute=0, second=0, microsecond=0)
    df = df.sort_index()
    matching = df.loc[df.index >= at_day]
    if matching.empty:
        return None
    close_value = matching["close"].iloc[0]
    if not math.isfinite(close_value):
        return None
    return float(close_value)


__all__ = [
    "CalibrationMaturationWorker",
    "CalibrationRecorder",
    "PriceLookup",
    "bars_cache_price_lookup",
    "derive_outcome",
]
