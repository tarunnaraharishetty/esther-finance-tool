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

export type SignalAction = "buy" | "sell" | "hold";

export type RecommendationTier =
  | "strong_buy"
  | "buy"
  | "hold"
  | "sell"
  | "strong_sell";

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
  level: "info" | "warn" | "error" | string;
  message: string;
  count: number;
}

export interface Alert {
  symbol: string;
  rule: string;
  severity: "info" | "warn" | "critical" | string;
  message: string;
  fired_at: string;
  // The Python dataclass has additional fields; we type the ones the
  // current panels render and leave the rest as a passthrough.
  [extra: string]: unknown;
}

export interface MarketPulse {
  // Categorical labels surfaced verbatim in the dashboard header.
  sentiment: string; // "bullish" | "bearish" | "neutral" | ...
  conviction: string; // "strong" | "weak" | ...
  activity: string; // "active" | "calm" | "intense" | ...
  summary: string; // pre-composed friendly headline
  // Numeric breadth + intensity fields used for the sparkline view.
  bullish_count: number;
  bearish_count: number;
  healthy_count: number;
  momentum_breadth: number;
  sentiment_breadth: number;
  reversal_intensity: number;
  alert_intensity: number;
  strongest_symbols: string[];
  [extra: string]: unknown;
}

export interface PulseEvolution {
  regime: "risk-on" | "risk-off" | "mixed" | "indeterminate" | string;
  patterns: string[];
  [extra: string]: unknown;
}

export interface OpportunityHistory {
  streak: number;
  appearances: number;
  window: number;
}

export interface DashboardSnapshot {
  tick: number;
  timestamp: string;
  rows: RecommendationRow[];
  events: EventEntry[];
  alerts: Alert[];
  recent_alerts: Alert[];
  signal_history: Record<string, unknown>;
  opp_history: Record<string, OpportunityHistory>;
  pulse: MarketPulse | null;
  pulse_history: unknown;
  pulse_evolution: PulseEvolution | null;
  intraday_signal_history: Record<string, unknown>;
  intraday_opp_history: Record<string, OpportunityHistory>;
  intraday_pulse: MarketPulse | null;
  intraday_pulse_history: unknown;
  intraday_pulse_evolution: PulseEvolution | null;
  session_store_status: unknown;
}
