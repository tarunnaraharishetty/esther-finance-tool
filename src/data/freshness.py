"""Centralized data-freshness policies.

Every consumer that produces a :class:`~src.data.envelope.DataEnvelope`
asks a policy here to grade the freshness of an ``as_of`` timestamp.
Policies live in one table so cross-cutting decisions ("what counts as
fresh news?", "when is an income statement too old to trust?") are
auditable in one place rather than scattered across the codebase.

Tier semantics
--------------
* ``fresh``    — fully trustworthy for analysis, alerts, AI claims.
* ``aging``    — still usable but should be visibly tagged in UI.
* ``stale``    — render for context only; do NOT drive probabilistic
  claims or alerts off it.
* ``expired``  — show with a strong warning or hide entirely;
  downstream features should refuse to score on it.

How tiers map
-------------
Given a policy ``(fresh_window, aging_window, expired_window)`` and a
data age ``age = now - as_of``:

* ``age <= fresh_window``     → ``fresh``
* ``age <= aging_window``     → ``aging``
* ``age <= expired_window``   → ``stale``
* ``age >  expired_window``   → ``expired``

The boundaries are inclusive so a record exactly at the ``fresh_window``
boundary is still ``fresh``.

Adding a new policy
-------------------
Pick a stable string key (``{domain}.{subtype}``), add an entry to
:data:`POLICIES`, and pass the key to :func:`policy_for` at the
consumer site. The key naming is mildly enforced by tests so typos
fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.data.envelope import Freshness


@dataclass(frozen=True)
class FreshnessPolicy:
    """Three-tier window boundaries for a data class.

    Windows must satisfy ``fresh_window < aging_window < expired_window``;
    the constructor validates this so a misconfigured policy fails at
    import time rather than at evaluation time.
    """

    fresh_window: timedelta
    aging_window: timedelta
    expired_window: timedelta

    def __post_init__(self) -> None:
        # Frozen dataclass; use object.__setattr__ would be needed for
        # mutation, but we're not mutating — just validating.
        if not (self.fresh_window < self.aging_window < self.expired_window):
            raise ValueError(
                "FreshnessPolicy windows must be strictly increasing: "
                f"fresh={self.fresh_window}, aging={self.aging_window}, "
                f"expired={self.expired_window}"
            )


# Canonical policy table. String keys follow the ``{domain}.{subtype}``
# convention so they read naturally at the call site. The windows are
# tuned to discretionary-trader expectations, not to algorithmic
# trading or microstructure analysis.
POLICIES: dict[str, FreshnessPolicy] = {
    # Quarterly fundamentals. Companies report every ~90 days; a 95-day
    # fresh window covers the typical filing cadence. 180 days = a full
    # quarter past, so the data is one earnings cycle stale. 400 days =
    # over a year — at that point we should refuse to score on it.
    "fundamentals.quarterly": FreshnessPolicy(
        fresh_window=timedelta(days=95),
        aging_window=timedelta(days=180),
        expired_window=timedelta(days=400),
    ),
    # Annual fundamentals. 10-K cadence is roughly yearly; a 400-day
    # fresh window absorbs late filings. Expired at two years — at that
    # point the business has materially changed.
    "fundamentals.annual": FreshnessPolicy(
        fresh_window=timedelta(days=400),
        aging_window=timedelta(days=550),
        expired_window=timedelta(days=730),
    ),
    # Intraday bars. Anything older than 90 seconds is no longer "live".
    # 5 minutes is still actionable for a slow trader; past 30 minutes
    # the bar should be treated as historical context.
    "bars.intraday": FreshnessPolicy(
        fresh_window=timedelta(seconds=90),
        aging_window=timedelta(minutes=5),
        expired_window=timedelta(minutes=30),
    ),
    # Daily bars. 25 hours covers an overnight + the next session's
    # open. After 3 days a daily bar is materially behind the market.
    "bars.daily": FreshnessPolicy(
        fresh_window=timedelta(hours=25),
        aging_window=timedelta(days=3),
        expired_window=timedelta(days=7),
    ),
    # News articles. 6 hours = a half-session of relevance. 24 hours =
    # a full session passed. After 3 days the news is historical, not
    # actionable.
    "news": FreshnessPolicy(
        fresh_window=timedelta(hours=6),
        aging_window=timedelta(hours=24),
        expired_window=timedelta(days=3),
    ),
    # Analyst price targets. Move slowly; 14 days is a reasonable
    # fresh window. After 4 months the consensus has likely drifted.
    "analyst_targets": FreshnessPolicy(
        fresh_window=timedelta(days=14),
        aging_window=timedelta(days=45),
        expired_window=timedelta(days=120),
    ),
}


class UnknownPolicyError(KeyError):
    """Raised when a caller asks for a policy key that isn't in :data:`POLICIES`."""


def policy_for(key: str) -> FreshnessPolicy:
    """Look up a policy by key. Raises :class:`UnknownPolicyError` on miss.

    Callers should fail loud rather than silently use a default policy
    — silent defaults are exactly how stale data gets through.
    """
    try:
        return POLICIES[key]
    except KeyError:
        raise UnknownPolicyError(
            f"Unknown freshness policy {key!r}. Known: {sorted(POLICIES)}"
        ) from None


def evaluate_freshness(
    as_of: datetime, now: datetime, policy: FreshnessPolicy
) -> Freshness:
    """Grade ``as_of`` against ``now`` per ``policy``.

    ``as_of`` in the future (provider clock skew) is treated as zero
    age — we never report negative-age data as anything but ``fresh``.
    """
    age = now - as_of
    if age.total_seconds() < 0:
        age = timedelta(0)
    if age <= policy.fresh_window:
        return "fresh"
    if age <= policy.aging_window:
        return "aging"
    if age <= policy.expired_window:
        return "stale"
    return "expired"


__all__ = [
    "POLICIES",
    "FreshnessPolicy",
    "UnknownPolicyError",
    "evaluate_freshness",
    "policy_for",
]
