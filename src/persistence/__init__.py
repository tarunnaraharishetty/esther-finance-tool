"""Cross-session persistence for the intelligence layer.

Saves and restores tracker state (signal history, OPP membership,
pulse history, alert log) across dashboard restarts so a mid-day
resume picks up where the session left off rather than starting
cold.

Live Alpaca data (bars, news) is NOT persisted here — those have
their own cache in :mod:`src.data.cache`. This package is strictly
for intelligence-layer trackers that derive from the per-tick
recommendations.
"""

from src.persistence.session_store import (
    PulseRecord,
    SessionSnapshot,
    SessionStore,
)

__all__ = ["PulseRecord", "SessionSnapshot", "SessionStore"]
