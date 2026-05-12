"""Intelligence layer: explanations, summaries, alerts, watchlist diffs.

This is the human-facing reasoning that sits between the raw scores
produced by :mod:`src.strategy.recommendation` and the dashboard UI.
Nothing here submits orders or makes autonomous decisions.
"""

from src.intelligence.alerts import (
    ActionChangedRule,
    Alert,
    AlertEngine,
    ConfidenceThresholdRule,
    Rule,
    SentimentShiftRule,
)
from src.intelligence.explain import Contributor, Explanation, explain
from src.intelligence.summary import Summarizer, TemplateSummarizer
from src.intelligence.watchlist import (
    WatchlistChange,
    action_breakdown,
    diff_snapshots,
    top_movers,
)

__all__ = [
    "ActionChangedRule",
    "Alert",
    "AlertEngine",
    "ConfidenceThresholdRule",
    "Contributor",
    "Explanation",
    "Rule",
    "SentimentShiftRule",
    "Summarizer",
    "TemplateSummarizer",
    "WatchlistChange",
    "action_breakdown",
    "diff_snapshots",
    "explain",
    "top_movers",
]
