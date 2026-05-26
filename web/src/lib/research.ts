/**
 * Research-thesis API client + React hook.
 *
 * Mirrors the dataclass shape in `src/intelligence/research_thesis.py`
 * one-to-one — keep them in sync. Runtime validation via zod (B-18)
 * means a backend schema drift surfaces as a clean error banner
 * naming the offending field rather than crashing mid-render with
 * ``Cannot read properties of undefined``.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";

import { validateJson } from "@/lib/_validate";
import type { TrustScore } from "@/lib/trust";

export type Rating =
  | "strong_buy"
  | "buy"
  | "hold"
  | "sell"
  | "strong_sell";

// Re-export the shared ProvenanceMap so existing imports keep working.
// The canonical definition lives in `@/lib/provenance` so the type
// can be reused by surfaces that don't depend on research (e.g.
// the compare narrative).
import type { ProvenanceMap } from "./provenance";

export type { ProvenanceMap };

export interface ThesisSection {
  title: string;
  body: string;
  bullets: string[];
  provenance: ProvenanceMap;
}

export interface BullBearArgument {
  label: string;
  weight: number; // 0..1
  detail: string;
  provenance: ProvenanceMap;
}

export interface Catalyst {
  label: string;
  when: string;
  impact: "bullish" | "bearish" | "uncertain";
  detail: string;
  provenance: ProvenanceMap;
}

export interface OutlookEntry {
  horizon: "short" | "medium" | "long";
  bias: "bullish" | "bearish" | "neutral";
  confidence: number; // 0..1
  detail: string;
  provenance: ProvenanceMap;
}

export interface MetricEntry {
  label: string;
  value: string;
  delta: string | null;
  tone: "bull" | "bear" | "warn" | null;
  provenance: ProvenanceMap;
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


// ---------------------------------------------------------------------------
// Zod schemas — runtime mirrors of the interfaces above (B-18).
// ---------------------------------------------------------------------------
//
// Keep these schemas in lockstep with the interfaces above AND with the
// pydantic dataclasses in `src/intelligence/research_thesis.py`. The
// frontend tests in `research.test.ts` (and any vitest covering this
// hook) drive a sample API response through ``ResearchThesisSchema``;
// a backend dataclass that drops or renames a field fails the schema
// parse with a single-line message naming the offending path.
//
// ``ProvenanceMap`` is a free-form ``Record<string, string>`` so the
// schema is ``z.record(z.string(), z.string())`` rather than a fixed
// object — the keys are token strings the validator emitted, not a
// stable enum.

const RatingSchema = z.enum([
  "strong_buy",
  "buy",
  "hold",
  "sell",
  "strong_sell",
]);

const ProvenanceMapSchema = z.record(z.string(), z.string());

const ThesisSectionSchema = z.object({
  title: z.string(),
  body: z.string(),
  bullets: z.array(z.string()),
  provenance: ProvenanceMapSchema,
});

const BullBearArgumentSchema = z.object({
  label: z.string(),
  weight: z.number(),
  detail: z.string(),
  provenance: ProvenanceMapSchema,
});

const CatalystSchema = z.object({
  label: z.string(),
  when: z.string(),
  impact: z.enum(["bullish", "bearish", "uncertain"]),
  detail: z.string(),
  provenance: ProvenanceMapSchema,
});

const OutlookEntrySchema = z.object({
  horizon: z.enum(["short", "medium", "long"]),
  bias: z.enum(["bullish", "bearish", "neutral"]),
  confidence: z.number(),
  detail: z.string(),
  provenance: ProvenanceMapSchema,
});

const MetricEntrySchema = z.object({
  label: z.string(),
  value: z.string(),
  delta: z.string().nullable(),
  tone: z.enum(["bull", "bear", "warn"]).nullable(),
  provenance: ProvenanceMapSchema,
});

const DroppedClaimSchema = z.object({
  section: z.string(),
  sentence: z.string(),
  unsupported_tokens: z.array(z.string()),
});

const ValidationReportSchema = z.object({
  drop_count: z.number(),
  dropped_claims: z.array(DroppedClaimSchema),
});

// Trust score lives in `@/lib/trust` as a TS interface. We mirror its
// wire shape inline so this module doesn't depend on a TrustScore
// schema export — the moment that lib grows its own zod schema, this
// can be swapped for `TrustScoreSchema` directly.
const TrustComponentStatusSchema = z.enum([
  "ok",
  "warn",
  "missing",
  "n/a",
]);

const TrustComponentSchema = z.object({
  name: z.string(),
  weight: z.number(),
  value: z.number().nullable(),
  contribution: z.number().nullable(),
  status: TrustComponentStatusSchema,
  detail: z.string(),
});

const TrustScoreSchema = z.object({
  score: z.number(),
  grade: z.enum(["A+", "A", "B", "C", "D", "F"]),
  components: z.array(TrustComponentSchema),
});

export const ResearchThesisSchema = z.object({
  symbol: z.string(),
  generated_at: z.string(),
  model: z.string(),
  rating: RatingSchema,
  confidence: z.number(),
  tagline: z.string(),
  company_overview: ThesisSectionSchema,
  bull_thesis: z.array(BullBearArgumentSchema),
  bear_thesis: z.array(BullBearArgumentSchema),
  technical_analysis: ThesisSectionSchema,
  fundamental_analysis: ThesisSectionSchema,
  metrics: z.array(MetricEntrySchema),
  sentiment_news: ThesisSectionSchema,
  catalysts: z.array(CatalystSchema),
  risk_assessment: ThesisSectionSchema,
  outlook: z.array(OutlookEntrySchema),
  explainability: ThesisSectionSchema,
  data_sources: z.array(z.string()),
  disclaimers: z.array(z.string()),
  cache: z.enum(["hit", "miss"]),
  cache_age_seconds: z.number(),
  mode: z.enum(["llm", "template"]),
  warning: z.string().optional(),
  validation: ValidationReportSchema,
  trust_score: TrustScoreSchema,
});

// Sanity-check the schema and the hand-written interface agree at
// compile time. If the interface adds a field the schema must too —
// the assignment fails the typecheck otherwise. Cheap insurance
// against the schema and the interface drifting independently.
type _SchemaMatchesInterface = z.infer<typeof ResearchThesisSchema> extends ResearchThesis ? true : false;
const _researchSchemaCheck: _SchemaMatchesInterface = true;
void _researchSchemaCheck;

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
        // Runtime schema validation (B-18). A backend dataclass drift
        // surfaces as a one-line error naming the offending field
        // instead of crashing mid-render. The compile-time assertion
        // above guarantees the schema + interface agree.
        const data = await validateJson(res, ResearchThesisSchema, "research");
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
