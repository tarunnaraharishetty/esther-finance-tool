"""Intelligence layer: explanations, summaries, alerts, watchlist diffs.

This is the human-facing reasoning that sits between the raw scores
produced by :mod:`src.strategy.recommendation` and the dashboard UI.
Nothing here submits orders or makes autonomous decisions.
"""

from src.intelligence.alert_prioritizer import (
    AlertPrioritizer,
    AlertState,
    CompositeRule,
    PrioritizerConfig,
)
from src.intelligence.alerts import (
    ActionChangedRule,
    Alert,
    AlertEngine,
    ConfidenceThresholdRule,
    Rule,
    SentimentShiftRule,
    load_prioritizer_config_from_yaml,
    load_rules_from_yaml,
)
from src.intelligence.explain import Contributor, Explanation, explain
from src.intelligence.history import SignalEpisode, SignalHistory, SignalHistorySummary
from src.intelligence.llm_summary import LLMSummarizer
from src.intelligence.pulse import MarketPulse, compute_pulse
from src.intelligence.recap import LLMRecapGenerator, RecapContext
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
    "AlertPrioritizer",
    "AlertState",
    "CompositeRule",
    "ConfidenceThresholdRule",
    "Contributor",
    "Explanation",
    "LLMRecapGenerator",
    "LLMSummarizer",
    "MarketPulse",
    "PrioritizerConfig",
    "RecapContext",
    "Rule",
    "SentimentShiftRule",
    "SignalEpisode",
    "SignalHistory",
    "SignalHistorySummary",
    "Summarizer",
    "TemplateSummarizer",
    "WatchlistChange",
    "action_breakdown",
    "compute_pulse",
    "diff_snapshots",
    "explain",
    "load_prioritizer_config_from_yaml",
    "load_rules_from_yaml",
    "top_movers",
]
