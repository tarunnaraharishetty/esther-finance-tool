"""Generic data freshness envelope.

Every fetched record in Esther flows through a :class:`DataEnvelope`.
The envelope carries the data plus a uniform freshness / attribution /
confidence contract so downstream consumers — the analyzer, the AI
research layer, the dashboard — never have to recompute "how old is
this" or "who told us this" or "how much should we trust the source".

The envelope is intentionally provider-agnostic. ``source_chain`` is
``tuple[str, ...]`` rather than a fundamentals-specific enum so the
same envelope can wrap bars, news, fundamentals, or anything else the
data layer fetches.

Why this matters
----------------
A core institutional trust property: a stale fact should be visibly
stale, not silently substituted for a fresh one. Without an envelope,
each consumer would have to reach into the underlying record, find a
timestamp field whose semantics differ per data class, and decide for
itself what "old" means. Centralizing that here makes the contract
auditable and the UI honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

Freshness = Literal["fresh", "aging", "stale", "expired"]


@dataclass(frozen=True)
class DataEnvelope[T]:
    """Wraps a piece of fetched data with its freshness contract.

    Attributes:
        data: The payload (any type).
        as_of: When the underlying fact was true (e.g. statement
            ``fiscal_date``, bar timestamp, article ``published_at``).
            Drives all freshness math.
        fetched_at: When we pulled the data from the provider. Used
            for cache-age reasoning and operator forensics; not for
            freshness scoring.
        source_chain: Names of providers we consulted, in order. The
            successful provider is the last entry; earlier entries
            represent attempted-and-failed upstreams. Opaque to this
            layer.
        freshness: One of ``fresh / aging / stale / expired`` per the
            policy that wrapped this envelope. See :mod:`src.data.freshness`.
        provider_confidence: ``[0, 1]`` confidence the *source* of the
            data is trustworthy. Distinct from analytical confidence
            (valuation / technical confidence describe the analysis;
            this describes the input quality).
    """

    data: T
    as_of: datetime
    fetched_at: datetime
    source_chain: tuple[str, ...]
    freshness: Freshness
    provider_confidence: float

    @property
    def data_age(self) -> timedelta:
        """How old the underlying fact is, measured at fetch time.

        Returns a non-negative ``timedelta``; if ``as_of`` is somehow
        in the future relative to ``fetched_at`` (clock skew on the
        provider side) we clamp to zero rather than expose a negative.
        """
        delta = self.fetched_at - self.as_of
        if delta.total_seconds() < 0:
            return timedelta(0)
        return delta

    @property
    def is_fresh(self) -> bool:
        return self.freshness == "fresh"

    @property
    def is_actionable(self) -> bool:
        """True when the envelope is fresh enough to drive a decision.

        ``aging`` data is still usable but should be visibly tagged.
        ``stale`` and ``expired`` data may be rendered for context but
        should not be the basis of a probabilistic claim or an alert.
        """
        return self.freshness in ("fresh", "aging")


__all__ = ["DataEnvelope", "Freshness"]
