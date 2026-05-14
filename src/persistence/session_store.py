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


CURRENT_SCHEMA_VERSION = 3
"""Bumped when the snapshot shape changes in a way that warrants
calling it out — even when the change is additive. Old snapshots
go through in-memory migration (pydantic defaults fill new
fields); any version we don't recognize cold-starts cleanly."""

_SUPPORTED_LOAD_VERSIONS: frozenset[int] = frozenset({1, 2, 3})
"""Versions ``load()`` knows how to read. v1 = pre-MT2; v2 added
``intraday_signal_episodes``; v3 added intraday opp tracker
fields. Pydantic defaults fill missing fields on load and the
controller re-persists at the current version."""


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

    # Brief caches — Anthropic-generated row + OPP briefs survive
    # restarts so the next ``s`` / ``b`` keypress doesn't re-bill the
    # LLM on a symbol the trader already paid for. Keys are tuples
    # in memory (``(symbol, action)`` / ``(symbol, composite_bucket)``);
    # JSON requires string dict keys so the controller joins with
    # ``|`` before serialization and splits on hydrate. Pydantic
    # defaults make these fields backward-compatible with snapshots
    # written before they existed.
    brief_cache: dict[str, str] = Field(default_factory=dict)
    opp_brief_cache: dict[str, str] = Field(default_factory=dict)

    # Intraday signal history (schema v2). Parallel to the daily
    # ``signal_episodes`` field above — recorded only when the
    # controller's ``intraday_enabled`` setting is on. v1 snapshots
    # land here as an empty dict, which is what we'd record for any
    # session that ran without the intraday feature.
    intraday_signal_episodes: dict[str, list[SignalEpisode]] = Field(default_factory=dict)

    # Intraday opportunity membership (schema v3). Parallel to the
    # daily ``opp_membership`` field — populated only when intraday
    # is enabled. Older snapshots load with an empty dict.
    intraday_opp_membership: dict[str, list[bool]] = Field(default_factory=dict)
    intraday_opp_window: int = 10

    # Intraday pulse history (MT2 phase 2c) — parallel to
    # ``pulse_records``. Additive; pydantic default keeps schema v3
    # snapshots loading without migration steps.
    intraday_pulse_records: list[PulseRecord] = Field(default_factory=list)
    intraday_pulse_window: int = 20


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
        if snapshot.schema_version not in _SUPPORTED_LOAD_VERSIONS:
            log.info(
                "session_store.schema_mismatch_starting_cold",
                path=str(self.path),
                snapshot_version=snapshot.schema_version,
                current_version=CURRENT_SCHEMA_VERSION,
            )
            return None
        if snapshot.schema_version != CURRENT_SCHEMA_VERSION:
            # In-memory migration: an older supported snapshot is
            # validated under the current model (pydantic fills new
            # fields with their defaults), and the controller will
            # re-persist it as the current version on the next tick.
            log.info(
                "session_store.migrated_in_memory",
                path=str(self.path),
                from_version=snapshot.schema_version,
                to_version=CURRENT_SCHEMA_VERSION,
            )
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
