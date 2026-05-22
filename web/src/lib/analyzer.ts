/**
 * Financial Analyzer API client + React hook.
 *
 * Mirrors the JSON shape produced by `src/api/analyzer.py:_serialize_report`
 * one-to-one. Adding a field on the backend without updating this file
 * will fail the strict tsc compile — the analyzer tab is the only
 * caller and it expects the typed surface.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// ---- citation vocabulary ----

export type CitationTag =
  | "technicals"
  | "fundamentals"
  | "news"
  | "sentiment"
  | "valuation"
  | "volume";

export const ALL_CITATION_TAGS: readonly CitationTag[] = [
  "technicals",
  "fundamentals",
  "news",
  "sentiment",
  "valuation",
  "volume",
] as const;

// ---- technical sub-scores ----

export interface TechnicalSubscores {
  rsi_overbought_score: number | null;
  rsi_oversold_score: number | null;
  bollinger_extension_score: number | null;
  ma_extension_score: number | null;
  volume_spike_score: number | null;
  atr_volatility_score: number | null;
  momentum_exhaustion_score: number | null;
}

export interface TechnicalScores {
  subscores: TechnicalSubscores;
  overbought_score: number | null;
  oversold_score: number | null;
  pullback_risk: number | null;
  rebound_potential: number | null;
  confidence_score: number;
  raw_rsi: number | null;
  raw_bollinger_z: number | null;
  raw_ma_distance_pct: number | null;
  raw_volume_z: number | null;
  raw_atr_ratio: number | null;
}

// ---- valuation ----

export interface ValuationEstimate {
  method: string;
  fair_value: number;
  confidence: number;
  inputs_used: string;
}

export interface ValuationEnsemble {
  estimates: ValuationEstimate[];
  bear_case: number | null;
  base_case: number | null;
  bull_case: number | null;
  weighted_ai_fair_value: number | null;
  confidence_score: number;
}

// ---- fundamentals profile ----

export interface FundamentalsProfile {
  symbol: string;
  name: string | null;
  sector: string | null;
  industry: string | null;
  exchange: string | null;
  country: string | null;
  currency: string;
  market_cap: number | null;
  enterprise_value: number | null;
  shares_outstanding: number | null;
  description: string | null;
  cik: string | null;
}

// ---- calibration (historical hit-rate lookup) ----

/**
 * One calibrated probability reading per ``(score, outcome)`` pairing
 * the analyzer publishes. Mirrors
 * ``src.intelligence.analyzer.calibration.CalibrationReading`` plus the
 * flattened bucket fields produced by the wire serializer.
 *
 * ``bucket_published`` is the single source of truth for whether the
 * UI should render a probability:
 *   * ``true``  → ``hit_rate`` + Wilson CI are populated; show the
 *                 calibrated probability with the observation count.
 *   * ``false`` → ``bucket_*`` and ``hit_rate`` are ``null``; show
 *                 "calibration pending" alongside the raw score value.
 *
 * Never compute a probability from absence of data. Never publish
 * ``hit_rate`` when ``bucket_published`` is false.
 */
export interface CalibrationReading {
  score_name: string;
  score_value: number;
  outcome_name: string;
  horizon_days: number;
  bucket_published: boolean;
  bucket_lo: number | null;
  bucket_hi: number | null;
  n_observations: number | null;
  n_hits: number | null;
  hit_rate: number | null;
  confidence_low: number | null;
  confidence_high: number | null;
  last_updated: string | null; // ISO-8601 with timezone
}

// ---- scenarios (probabilistic price-range model) ----

/**
 * Closed-form lognormal scenario model from
 * ``src.intelligence.analyzer.scenarios.ScenarioModel``. Always
 * ``null`` when bars or last_price are missing — the analyzer card
 * should render "scenarios unavailable" without complaint.
 *
 * Framing rule: probabilities here describe *current dispersion*, not
 * a forecast. The component label must be "model-implied probability",
 * never "chance of" or "likelihood that" — the latter framings imply
 * a directional view we don't have (μ = 0 by design).
 */
export interface ScenarioModel {
  horizon_days: number;
  current_price: number;
  annualized_vol: number;
  vol_lookback_days: number;
  vol_method: string;
  prob_below_bear: number | null;
  prob_above_bull: number | null;
  quantile_20: number;
  quantile_50: number;
  quantile_80: number;
  notes: string[];
}

// ---- freshness envelope ----

/**
 * Freshness tier for a data record. Mirrors
 * ``src.data.envelope.Freshness``. The UI keys staleness badges off
 * this enum:
 *  * fresh   → fully trustworthy
 *  * aging   → usable, render a soft "aging" badge
 *  * stale   → render with a warning; downstream alerts must skip
 *  * expired → strong warning or hide entirely
 */
export type Freshness = "fresh" | "aging" | "stale" | "expired";

/**
 * Freshness envelope attached to fundamentals on the analyzer report.
 * Always ``null`` when fundamentals are unavailable (chain exhaustion
 * or no providers configured).
 *
 * ``divergence_count`` is the number of high-trust fields where the
 * primary provider disagreed with the reconciliation secondary by
 * more than the configured threshold. Non-zero values should drive a
 * visible "providers disagree" indicator. ``reconciliation_warning_count``
 * carries notes about *why* a field wasn't compared (typically
 * fiscal-date mismatch); informational, not a data-quality flag.
 */
export interface FundamentalsFreshness {
  as_of: string; // ISO-8601 with timezone
  fetched_at: string; // ISO-8601 with timezone
  freshness: Freshness;
  data_age_days: number;
  source_chain: string[]; // provider names in the order they were tried
  provider_confidence: number; // [0, 1]
  divergence_count: number;
  reconciliation_warning_count: number;
}

// ---- explanation ----

export interface AnalyzerExplanation {
  symbol: string;
  generated_at: string;
  model: string;
  summary: string;
  claims: string[];
  risk_warnings: string[];
  citations: Partial<Record<CitationTag, string[]>>;
}

// ---- top-level report ----

export interface AnalyzerReport {
  symbol: string;
  generated_at: string;
  last_price: number | null;
  overall_analyzer_score: number | null;
  technical_score: number | null;
  fundamental_score: number | null;
  valuation_score: number | null;
  technicals: TechnicalScores | null;
  valuation: ValuationEnsemble | null;
  fundamentals_profile: FundamentalsProfile | null;
  fundamentals_freshness: FundamentalsFreshness | null;
  scenarios: ScenarioModel | null;
  calibrations: CalibrationReading[];
  explanation: AnalyzerExplanation;
  warnings: string[];
  cache: "hit" | "miss";
  cache_age_seconds: number;
}

// ---- hook ----

interface UseAnalyzerReport {
  report: AnalyzerReport | null;
  loading: boolean;
  error: string | null;
  refresh: (opts?: { fresh?: boolean }) => Promise<void>;
}

/**
 * Fetch + cache the Financial Analyzer report for one symbol.
 *
 * Re-fetches when the symbol changes. ``refresh({ fresh: true })``
 * forces the backend to rebuild from upstream rather than serving
 * the on-disk cached blob.
 */
export function useAnalyzerReport(symbol: string | null): UseAnalyzerReport {
  const [report, setReport] = useState<AnalyzerReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);

  const fetchFor = useCallback(
    async (sym: string, opts?: { fresh?: boolean }): Promise<void> => {
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const url = `/api/analyzer/${encodeURIComponent(sym)}${
          opts?.fresh ? "?fresh=true" : ""
        }`;
        const res = await fetch(url, { credentials: "same-origin" });
        if (!res.ok) {
          let detail: string;
          try {
            const body = await res.json();
            detail = typeof body?.detail === "string" ? body.detail : res.statusText;
          } catch {
            detail = res.statusText;
          }
          throw new Error(`${res.status} ${detail}`);
        }
        const data = (await res.json()) as AnalyzerReport;
        if (reqRef.current === reqId) {
          setReport(data);
        }
      } catch (e) {
        if (reqRef.current === reqId) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (reqRef.current === reqId) {
          setLoading(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    if (symbol === null) {
      setReport(null);
      return;
    }
    void fetchFor(symbol);
  }, [symbol, fetchFor]);

  const refresh = useCallback(
    async (opts?: { fresh?: boolean }): Promise<void> => {
      if (symbol === null) return;
      await fetchFor(symbol, opts);
    },
    [symbol, fetchFor],
  );

  return { report, loading, error, refresh };
}

// ---- formatting helpers (kept here so component code stays presentational) ----

/** Pull tags off the end of a citation-tagged claim. */
export function parseClaimCitations(claim: string): {
  text: string;
  tags: CitationTag[];
} {
  const matches = [...claim.matchAll(/\[([a-z, ]+)\]/g)];
  if (matches.length === 0) return { text: claim, tags: [] };
  const tags: CitationTag[] = [];
  for (const m of matches) {
    for (const raw of m[1]!.split(",")) {
      const t = raw.trim();
      if (isCitationTag(t)) tags.push(t);
    }
  }
  // Strip the citation suffix(es) from the displayed text.
  const text = claim.replace(/\s*\[[a-z, ]+\]/g, "").trim();
  return { text, tags };
}

function isCitationTag(value: string): value is CitationTag {
  return (ALL_CITATION_TAGS as readonly string[]).includes(value);
}

/** Format a [0,100] score with sane fallback. */
export function fmtScore(score: number | null | undefined): string {
  if (score === null || score === undefined || !Number.isFinite(score)) {
    return "—";
  }
  return Math.round(score).toString();
}

/** Color tone bucket for a directional [0,100] score. */
export function scoreTone(
  score: number | null | undefined,
): "bull" | "bear" | "warn" | "muted" {
  if (score === null || score === undefined || !Number.isFinite(score)) {
    return "muted";
  }
  if (score >= 70) return "warn";
  if (score >= 40) return "bull";
  if (score >= 20) return "muted";
  return "muted";
}
