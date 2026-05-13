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
    OpportunityEntryRule,
    Rule,
    SentimentShiftRule,
    SnapshotRule,
    load_prioritizer_config_from_yaml,
    load_rules_from_yaml,
    load_snapshot_rules_from_yaml,
)
from src.intelligence.explain import Contributor, Explanation, explain
from src.intelligence.history import SignalEpisode, SignalHistory, SignalHistorySummary
from src.intelligence.llm_summary import LLMSummarizer
from src.intelligence.opportunities import (
    Opportunity,
    RankedOpportunity,
    detect_opportunities,
    rank_opportunities,
)
from src.intelligence.opportunity_brief import (
    LLMOpportunityBriefer,
    OpportunityBriefContext,
)
from src.intelligence.opportunity_drilldown import (
    DriverBreakdown,
    OpportunityDrilldown,
    build_drilldown,
)
from src.intelligence.opportunity_history import (
    OpportunityHistory,
    OpportunityMembershipTracker,
)
from src.intelligence.pulse import MarketPulse, compute_pulse
from src.intelligence.pulse_history import PulseHistory, PulseHistoryTracker
from src.intelligence.recap import LLMRecapGenerator, RecapContext
from src.intelligence.signal_profile import SignalProfile, compute_signal_profile
from src.intelligence.summary import Summarizer, TemplateSummarizer
from src.intelligence.tier import promote_to_tier
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
    "DriverBreakdown",
    "Explanation",
    "LLMOpportunityBriefer",
    "LLMRecapGenerator",
    "LLMSummarizer",
    "MarketPulse",
    "Opportunity",
    "OpportunityBriefContext",
    "OpportunityDrilldown",
    "OpportunityEntryRule",
    "OpportunityHistory",
    "OpportunityMembershipTracker",
    "PrioritizerConfig",
    "PulseHistory",
    "PulseHistoryTracker",
    "RankedOpportunity",
    "RecapContext",
    "Rule",
    "SentimentShiftRule",
    "SignalEpisode",
    "SignalHistory",
    "SignalHistorySummary",
    "SignalProfile",
    "SnapshotRule",
    "Summarizer",
    "TemplateSummarizer",
    "WatchlistChange",
    "action_breakdown",
    "build_drilldown",
    "compute_pulse",
    "compute_signal_profile",
    "detect_opportunities",
    "diff_snapshots",
    "explain",
    "load_prioritizer_config_from_yaml",
    "load_rules_from_yaml",
    "load_snapshot_rules_from_yaml",
    "promote_to_tier",
    "rank_opportunities",
    "top_movers",
]
