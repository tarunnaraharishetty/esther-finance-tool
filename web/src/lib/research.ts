/**
 * Research-thesis API client + React hook.
 *
 * Mirrors the dataclass shape in `src/intelligence/research_thesis.py`
 * one-to-one — keep them in sync. No passthrough escape hatches so
 * a backend schema drift fails at `tsc --strict` time instead of
 * mid-render.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { TrustScore } from "@/lib/trust";

export type Rating =
  | "strong_buy"
  | "buy"
  | "hold"
  | "sell"
  | "strong_sell";

export interface ThesisSection {
  title: string;
  body: string;
  bullets: string[];
}

export interface BullBearArgument {
  label: string;
  weight: number; // 0..1
  detail: string;
}

export interface Catalyst {
  label: string;
  when: string;
  impact: "bullish" | "bearish" | "uncertain";
  detail: string;
}

export interface OutlookEntry {
  horizon: "short" | "medium" | "long";
  bias: "bullish" | "bearish" | "neutral";
  confidence: number; // 0..1
  detail: string;
}

export interface MetricEntry {
  label: string;
  value: string;
  delta: string | null;
  tone: "bull" | "bear" | "warn" | null;
}

/**
 * One claim the post-hoc validator dropped from the rendered thesis.
 *
 * ``section`` identifies the field path on the thesis ("technical_analysis.body",
 * "bull_thesis[0].detail", "metrics[2].value"); ``sentence`` is the
 * dropped text verbatim; ``unsupported_tokens`` lists the numeric
 * tokens that didn't anchor in the input bundle.
 */
export interface DroppedClaim {
  section: string;
  sentence: string;
  unsupported_tokens: string[];
}

/**
 * Validator output attached to every research response. ``drop_count``
 * is the load-bearing field — the badge keys off it. A non-zero
 * count is a *positive* trust signal: "AI declined to make N
 * unsupported claims". The UI must never hide or soften it.
 */
export interface ValidationReport {
  drop_count: number;
  dropped_claims: DroppedClaim[];
}

export interface ResearchThesis {
  symbol: string;
  generated_at: string; // ISO-8601
  model: string;
  rating: Rating;
  confidence: number; // 0..1
  tagline: string;
  company_overview: ThesisSection;
  bull_thesis: BullBearArgument[];
  bear_thesis: BullBearArgument[];
  technical_analysis: ThesisSection;
  fundamental_analysis: ThesisSection;
  metrics: MetricEntry[];
  sentiment_news: ThesisSection;
  catalysts: Catalyst[];
  risk_assessment: ThesisSection;
  outlook: OutlookEntry[];
  explainability: ThesisSection;
  data_sources: string[];
  disclaimers: string[];
  // Endpoint-attached metadata.
  cache: "hit" | "miss";
  cache_age_seconds: number;
  mode: "llm" | "template";
  warning?: string;
  // Always present (empty array when nothing dropped).
  validation: ValidationReport;
  // Composite report trust grade. Mirrors `src/intelligence/trust_score.py`.
  trust_score: TrustScore;
}

interface UseResearchThesis {
  thesis: ResearchThesis | null;
  loading: boolean;
  error: string | null;
  refresh: (opts?: { fresh?: boolean }) => Promise<void>;
}

/**
 * Fetch + cache a thesis for one symbol.
 *
 * Re-fetches when the symbol changes. `refresh({ fresh: true })`
 * asks the backend to bypass its cache and regenerate.
 */
export function useResearchThesis(symbol: string | null): UseResearchThesis {
  const [thesis, setThesis] = useState<ResearchThesis | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // Tracks the active fetch so a stale response doesn't clobber a fresher one.
  const reqRef = useRef<number>(0);

  const fetchFor = useCallback(
    async (sym: string, opts?: { fresh?: boolean }): Promise<void> => {
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const url = `/api/research/${encodeURIComponent(sym)}${
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
        const data = (await res.json()) as ResearchThesis;
        if (reqRef.current === reqId) {
          setThesis(data);
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
      setThesis(null);
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

  return { thesis, loading, error, refresh };
}

/** Human-readable label for the rating enum. */
export function ratingLabel(rating: Rating): string {
  return rating.replace(/_/g, " ").toUpperCase();
}

/** Tone-mapping for badges / colored UI elements. */
export function ratingTone(rating: Rating): "bull" | "bear" | "warn" {
  if (rating === "strong_buy" || rating === "buy") return "bull";
  if (rating === "strong_sell" || rating === "sell") return "bear";
  return "warn";
}
