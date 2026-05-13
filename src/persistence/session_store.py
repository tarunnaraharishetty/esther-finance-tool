"""JSON-backed session-state persistence.

One snapshot file per installation (path configurable via
``Settings.session_state_path``). The controller hydrates trackers
from it at startup and writes the current state back at the end of
every tick.

Write is atomic — content goes to ``<path>.tmp`` first, then renames
over the canonical path. A crash mid-write never leaves the
canonical file partially written; the prior snapshot survives.

Read is corruption-resilient — a missing file, invalid JSON, or
unknown ``schema_version`` returns ``None`` (logged warning) and the
caller starts from a cold tracker rather than crashing.

Schema is captured by :class:`SessionSnapshot`. Incompatible changes
bump ``CURRENT_SCHEMA_VERSION`` and old snapshots are silently
discarded on next load.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.intelligence.alerts import Alert
from src.intelligence.history import SignalEpisode
from src.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


CURRENT_SCHEMA_VERSION = 1
"""Bumped on incompatible schema changes. Old snapshots are
discarded by :meth:`SessionStore.load` rather than migrated — the
intelligence trackers self-heal on first tick, so the cost of a
cold start after a schema bump is one missed warm-up tick."""


class PulseRecord(BaseModel):
    """One persisted pulse tick.

    Captures every field the live :class:`~src.intelligence.pulse.MarketPulse`
    carries so the rolling history round-trips losslessly. Stored as a
    plain pydantic model rather than the original dataclass because
    pydantic handles JSON serialization out of the box.
    """

    model_config = ConfigDict(frozen=True)

    sentiment: str
    conviction: str
    activity: str
    summary: str
    bullish_count: int = 0
    bearish_count: int = 0
    healthy_count: int = 0
    momentum_breadth: float = 0.0
    sentiment_breadth: float = 0.0
    reversal_intensity: int = 0
    alert_intensity: int = 0
    strongest_symbols: list[tuple[str, str]] = Field(default_factory=list)


class SessionSnapshot(BaseModel):
    """Top-level persisted state.

    Every field has a default so a freshly-constructed snapshot
    serializes cleanly even before any data lands. ``schema_version``
    is read on load to drop snapshots written by an incompatible
    version of the codebase.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = CURRENT_SCHEMA_VERSION
    saved_at: datetime
    tick: int = 0

    # Signal history — per-symbol episode log.
    signal_episodes: dict[str, list[SignalEpisode]] = Field(default_factory=dict)
    signal_history_max: int = 10

    # Opportunity membership — per-symbol rolling boolean buffer.
    opp_membership: dict[str, list[bool]] = Field(default_factory=dict)
    opp_window: int = 10

    # Pulse history — rolling list of pulse records.
    pulse_records: list[PulseRecord] = Field(default_factory=list)
    pulse_window: int = 20

    # Alert state — bounded log + last-fired map.
    alert_log: list[Alert] = Field(default_factory=list)
    alert_last_fired: dict[str, datetime] = Field(default_factory=dict)
    alert_max_history: int = 200


class SessionStore:
    """JSON-file persistence for :class:`SessionSnapshot`.

    Construct with a path; call :meth:`load` once at startup and
    :meth:`save` once per tick. Both operations are best-effort —
    failures are logged but never raised, because losing a session
    snapshot is not a reason to crash the trader's dashboard.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> SessionSnapshot | None:
        """Read the snapshot if present, valid, and schema-compatible.

        Returns ``None`` on missing file, invalid JSON, validation
        error, or schema-version mismatch — every failure case looks
        the same to the caller and they start cold.
        """
        if not self.path.exists():
            return None
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("session_store.read_failed", path=str(self.path), error=str(e))
            return None
        try:
            snapshot = SessionSnapshot.model_validate_json(text)
        except ValidationError as e:
            log.warning(
                "session_store.invalid_snapshot",
                path=str(self.path),
                error=str(e),
            )
            return None
        if snapshot.schema_version != CURRENT_SCHEMA_VERSION:
            log.info(
                "session_store.schema_mismatch_starting_cold",
                path=str(self.path),
                snapshot_version=snapshot.schema_version,
                current_version=CURRENT_SCHEMA_VERSION,
            )
            return None
        return snapshot

    def save(self, snapshot: SessionSnapshot) -> None:
        """Atomic write — serialize to a sibling ``.tmp`` then rename.

        Best-effort: parent-dir creation, write, rename, all swallow
        OSError into a logged warning. The dashboard keeps running on
        a failed persist; the next tick will retry.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning(
                "session_store.mkdir_failed",
                path=str(self.path.parent),
                error=str(e),
            )
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(snapshot.model_dump_json(), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            log.warning(
                "session_store.write_failed",
                path=str(self.path),
                error=str(e),
            )
            # Best-effort cleanup of the orphaned tmp file.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "PulseRecord",
    "SessionSnapshot",
    "SessionStore",
]
