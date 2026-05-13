"""Pure-data ranked sections for the watchlist panel.

Given a :class:`~src.dashboard.state.DashboardSnapshot`, produce small
ranked lists the dashboard can render directly (and that ``recap.py``
feeds into the LLM context). No LLM, no state, no caching — recompute
every tick.

Each ranking returns a list of ``(symbol, score)`` tuples, capped at
``n`` (default 3), highest-magnitude first. Error rows are excluded
everywhere — they can't be ranked meaningfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.intelligence.history import SignalHistorySummary


@dataclass(frozen=True)
class Rankings:
    """All six ranked sections for one snapshot.

    Each list is most-extreme-first, capped at ``n`` per the caller's
    ``compute(snap, n=...)``. Lists can be empty if the snapshot has no
    healthy rows.
    """

    strongest_momentum: tuple[tuple[str, float], ...]
    strongest_sentiment: tuple[tuple[str, float], ...]
    biggest_reversals: tuple[tuple[str, float], ...]
    highest_confidence: tuple[tuple[str, float], ...]
    unusual_movers: tuple[tuple[str, float], ...]
    most_volatile: tuple[tuple[str, float], ...]


def compute(
    snapshot: DashboardSnapshot,
    *,
    n: int = 3,
) -> Rankings:
    """Compute all six ranked sections for the given snapshot.

    ``unusual_movers`` and ``most_volatile`` lean on per-symbol signal
    history when it's present on the snapshot. The other four rank rows
    directly without history.
    """
    healthy = [r for r in snapshot.rows if not r.error]

    return Rankings(
        strongest_momentum=_top_n(
            ((r.symbol, _macd_score(r)) for r in healthy),
            n=n,
            key_abs=True,
        ),
        strongest_sentiment=_top_n(
            ((r.symbol, r.sentiment_score) for r in healthy if r.num_news_articles > 0),
            n=n,
            key_abs=True,
        ),
        biggest_reversals=_top_n(
            _reversal_scores(healthy, snapshot.signal_history),
            n=n,
            key_abs=True,
        ),
        highest_confidence=_top_n(
            ((r.symbol, r.confidence) for r in healthy),
            n=n,
            key_abs=False,  # confidence is already non-negative
        ),
        unusual_movers=_top_n(
            _unusual_scores(healthy, snapshot.signal_history),
            n=n,
            key_abs=True,
        ),
        most_volatile=_top_n(
            _volatility_scores(healthy, snapshot.signal_history),
            n=n,
            key_abs=False,  # already non-negative (an "amount of variance")
        ),
    )


# ---------------------------------------------------------------------------
# Per-section score functions
# ---------------------------------------------------------------------------


def _macd_score(r: RecommendationRow) -> float:
    """MACD score (NaN-safe). MACD is our cleanest momentum proxy — the
    line tracks fast-vs-slow EMA so positive = upward momentum."""
    return 0.0 if r.macd != r.macd else r.macd  # NaN check


def _reversal_scores(
    rows: list[RecommendationRow],
    history: dict[str, SignalHistorySummary],
) -> "list[tuple[str, float]]":
    """A 'reversal' = the symbol's action flipped recently AND the
    direction implied by the prior episode is the opposite of the
    current.

    Score sign carries direction: positive = bearish-to-bullish flip,
    negative = bullish-to-bearish flip. Magnitude = current confidence.
    """
    out: list[tuple[str, float]] = []
    for r in rows:
        summary = history.get(r.symbol)
        if summary is None or not summary.recent:
            continue
        prior = summary.recent[0]
        current_dir = _action_direction(r.action)
        prior_dir = _action_direction(prior.action)
        if current_dir == 0 or prior_dir == 0:
            continue
        if current_dir == prior_dir:
            continue
        # Sign indicates the direction of the flip (current's sign).
        out.append((r.symbol, current_dir * r.confidence))
    return out


def _unusual_scores(
    rows: list[RecommendationRow],
    history: dict[str, SignalHistorySummary],
) -> "list[tuple[str, float]]":
    """'Unusual' = the symbol has been in its current state for a long
    time and the confidence is changing fast. The score is
    ``|confidence_delta| * tick_count`` — rewards both stickiness and
    swing.

    Without history (first tick), nothing qualifies.
    """
    out: list[tuple[str, float]] = []
    for r in rows:
        summary = history.get(r.symbol)
        if summary is None:
            continue
        ep = summary.current
        delta = abs(ep.confidence_last - ep.confidence_first)
        # First-tick episodes have delta=0; they fall out naturally.
        score = delta * ep.tick_count
        if score > 0:
            out.append((r.symbol, score))
    return out


def _volatility_scores(
    rows: list[RecommendationRow],
    history: dict[str, SignalHistorySummary],
) -> "list[tuple[str, float]]":
    """Volatility proxy = number of distinct episodes for the symbol in
    this session. More flips = more volatile.

    Without history, all rows score 0 and are filtered out.
    """
    out: list[tuple[str, float]] = []
    for r in rows:
        summary = history.get(r.symbol)
        if summary is None:
            continue
        # current + recent gives total episode count
        episode_count = 1 + len(summary.recent)
        if episode_count > 1:
            out.append((r.symbol, float(episode_count)))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action_direction(action: object) -> int:
    """+1 bullish, -1 bearish, 0 neutral."""
    from src.strategy.base import SignalAction

    if action == SignalAction.BUY:
        return 1
    if action == SignalAction.SELL:
        return -1
    return 0


def _top_n(
    items: "object",
    *,
    n: int,
    key_abs: bool,
) -> tuple[tuple[str, float], ...]:
    """Sort ``items`` (an iterable of ``(symbol, score)``) by score
    descending and take the top ``n``.

    ``key_abs=True`` sorts by absolute magnitude (extremes-first) — used
    for signed scores. ``False`` sorts by raw value (largest-first) —
    used for non-negative scores like confidence.
    """
    materialized = list(items)  # type: ignore[arg-type]
    key = (lambda kv: abs(kv[1])) if key_abs else (lambda kv: kv[1])
    materialized.sort(key=key, reverse=True)
    return tuple(materialized[:n])


__all__ = ["Rankings", "compute"]
