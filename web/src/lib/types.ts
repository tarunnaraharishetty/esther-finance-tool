// TypeScript shapes mirroring the JSON the Esther API emits.
//
// These are NOT auto-generated. The backend currently ships no
// OpenAPI response models (per the create_app docstring) — the
// DashboardSnapshot dataclass is the informal contract. When we
// pin Pydantic response models in a later phase, an OpenAPI-typed
// client can replace this file.
//
// Field nullability follows the dataclass defaults: anything the
// controller may not populate (intraday triad, session-store status,
// pulse on a fixture-built snapshot) is typed `null`.
//
// **No passthrough escape hatches.** Every interface here enumerates
// the fields the backend actually emits, full stop. The previous
// version used ``[extra: string]: unknown`` on several types, which
// let server-side schema drift slip past ``tsc --strict`` and only
// surface as runtime React errors. The trade is that a backend field
// addition now requires a parallel TS edit — we'd rather have the
// compile-time fail than the white screen.

export type SignalAction = "buy" | "sell" | "hold";

export type RecommendationTier =
  | "strong_buy"
  | "buy"
  | "hold"
  | "sell"
  | "strong_sell";

export type EventLevel = "info" | "warn" | "error";

// Alert severity is YAML-configurable in ``config/alerts.yaml``, so
// operators can introduce arbitrary values. The standard set the
// engine ships with is enumerated; the ``string`` fallback covers
// the open extension point. CSS keys off the standard set; unknown
// values fall through to the default style.
export type AlertSeverity = "info" | "warn" | "critical" | (string & {});

export type PulseRegime = "risk-on" | "risk-off" | "mixed" | "indeterminate";

export interface IntradayRead {
  action: SignalAction;
  confidence: number;
  technical_score: number;
  combined_score: number;
}

export interface RecommendationRow {
  symbol: string;
  action: SignalAction;
  confidence: number;
  combined_score: number;
  technical_score: number;
  sentiment_score: number;
  rsi: number;
  macd: number;
  bollinger: number;
  last_price: number;
  num_news_articles: number;
  reasoning: string;
  timestamp: string; // ISO-8601 with timezone
  headlines: string[];
  tier: RecommendationTier;
  signal_quality: string;
  stability: string;
  quality_reasons: string[];
  intraday: IntradayRead | null;
  error: string | null;
}

export interface EventEntry {
  timestamp: string;
  level: EventLevel;
  message: string;
  count: number;
}

export interface Alert {
  symbol: string;
  rule: string;
  severity: AlertSeverity;
  message: string;
  fired_at: string; // ISO-8601 with timezone
}

// Mirror of MarketPulse from src/intelligence/pulse.py. The numeric
// breadth + intensity fields feed the metrics grid; the categorical
// triad (sentiment / conviction / activity) is rendered verbatim;
// ``summary`` is pre-composed by the engine.
export interface MarketPulse {
  sentiment: string; // engine emits "bullish" | "bearish" | "neutral" | "mixed"
  conviction: string; // "strong" | "weak" | "moderate"
  activity: string; // "calm" | "active" | "volatile" | "intense"
  summary: string;
  bullish_count: number;
  bearish_count: number;
  healthy_count: number;
  momentum_breadth: number;
  sentiment_breadth: number;
  reversal_intensity: number;
  alert_intensity: number;
  // Python source: ``tuple[tuple[str, str], ...]`` — each entry is
  // (symbol, descriptor). jsonable_encoder converts the outer tuple
  // to an array but the inner pair-tuple becomes a 2-element array
  // too. Type the pair as a tuple so a misuse like ``s[2]`` is a
  // compile error.
  strongest_symbols: Array<[string, string]>;
}

export interface TrajectoryPattern {
  name: string;
  label: string; // human-readable, ready to render verbatim
  detail: string; // short context phrase (e.g. "across last 6 ticks")
}

export interface PulseEvolution {
  regime: PulseRegime;
  patterns: TrajectoryPattern[];
}

// Rolling-window pulse history. All series arrays are aligned by
// index and length ≤ window. Used for sparkline rendering.
export interface PulseHistory {
  momentum_breadth: number[];
  sentiment_breadth: number[];
  bullish_count: number[];
  bearish_count: number[];
  reversal_intensity: number[];
  alert_intensity: number[];
  sentiment: string[];
  conviction: string[];
  activity: string[];
  window: number;
}

export interface OpportunityHistory {
  streak: number;
  appearances: number;
  window: number;
}

export interface SignalProfile {
  stability: string; // "stable" | "noisy"
  trend: string; // "strengthening" | "weakening" | "flat"
  persistence: string; // "persistent" | "flipping"
}

export interface RankedOpportunity {
  symbol: string;
  tier: RecommendationTier;
  composite_score: number;
  profile: SignalProfile;
  rationale: string[];
  technical_alignment: number;
  sentiment_alignment: number;
  confidence_acceleration: number;
  momentum_persistence: number;
  unusual_activity: number;
  reversal_strength: number;
  signal_quality_score: number;
}

export interface SignalEpisode {
  action: SignalAction;
  started_at: string; // ISO-8601
  last_seen_at: string;
  tick_count: number;
  confidence_first: number;
  confidence_last: number;
}

export interface SignalHistorySummary {
  current: SignalEpisode;
  recent: SignalEpisode[];
}

// Health snapshot of the persistent SessionStore. ``null`` on the
// snapshot when persistence is disabled (the typical ``--mock`` /
// test-fixture case). Mirrors src/persistence/session_store.py.
export interface SessionStoreStatus {
  health: "ok" | "stale" | "degraded" | (string & {});
  last_success_at: string | null;
  last_error_at: string | null;
  last_error: string | null;
  bytes: number | null;
  path: string; // Python Path → string after jsonable_encoder
}

export interface DashboardSnapshot {
  tick: number;
  timestamp: string;
  rows: RecommendationRow[];
  events: EventEntry[];
  alerts: Alert[];
  recent_alerts: Alert[];
  signal_history: Record<string, SignalHistorySummary>;
  opp_history: Record<string, OpportunityHistory>;
  ranked_opportunities: RankedOpportunity[];
  pulse: MarketPulse | null;
  pulse_history: PulseHistory | null;
  pulse_evolution: PulseEvolution | null;
  intraday_signal_history: Record<string, SignalHistorySummary>;
  intraday_opp_history: Record<string, OpportunityHistory>;
  intraday_ranked_opportunities: RankedOpportunity[];
  intraday_pulse: MarketPulse | null;
  intraday_pulse_history: PulseHistory | null;
  intraday_pulse_evolution: PulseEvolution | null;
  session_store_status: SessionStoreStatus | null;
}
