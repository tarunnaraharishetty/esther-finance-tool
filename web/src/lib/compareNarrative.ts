/**
 * Comparison-narrative API client + React hook.
 *
 * Mirrors the wire shape produced by `src/api/compare.py`'s
 * `/api/compare/{a}/{b}/narrative` route and the underlying
 * `src/intelligence/comparison_narrative.py` dataclasses.
 *
 * Unlike `useCompareView`, this hook does NOT fetch on mount —
 * narratives are LLM-expensive and the user opts in by clicking
 * "Generate AI narrative" on the page. The hook exposes a
 * ``trigger`` callable plus the standard loading/error/data state.
 */

import { useCallback, useRef, useState } from "react";

import type { TrustScore } from "@/lib/trust";

export type NarrativeMode = "template" | "llm" | "auto";

export interface NarrativeSection {
  title: string;
  body: string;
  bullets: string[];
}

export interface DroppedNarrativeClaim {
  section: string;
  sentence: string;
  unsupported_tokens: string[];
}

export interface NarrativeValidation {
  drop_count: number;
  dropped_claims: DroppedNarrativeClaim[];
}

export interface ComparisonNarrative {
  left_symbol: string;
  right_symbol: string;
  tagline: string;
  headline: NarrativeSection;
  momentum: NarrativeSection;
  valuation: NarrativeSection;
  risk: NarrativeSection;
  quality: NarrativeSection;
  bottom_line: NarrativeSection;
  model: string;
  generated_at: string; // ISO-8601
  warnings: string[];
  validation: NarrativeValidation;
  trust_score: TrustScore;
  mode: NarrativeMode;
  cache: "hit" | "miss";
  warning?: string;
  assembly_warnings: string[];
}

interface UseComparisonNarrative {
  narrative: ComparisonNarrative | null;
  loading: boolean;
  error: string | null;
  trigger: (opts?: { mode?: NarrativeMode; fresh?: boolean }) => Promise<void>;
  reset: () => void;
}

/**
 * Manual-trigger comparison narrative hook.
 *
 * The hook never auto-fetches on symbol change — the user must click
 * Generate to start. When either symbol is null, ``trigger`` is a
 * no-op (still safe to call). Switching symbols clears the previous
 * narrative so the user doesn't see stale prose from a different pair.
 */
export function useComparisonNarrative(
  left: string | null,
  right: string | null,
): UseComparisonNarrative {
  const [narrative, setNarrative] = useState<ComparisonNarrative | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);
  const lastPairRef = useRef<string | null>(null);

  // Detect symbol-pair changes and clear stale narrative.
  const pairKey = left !== null && right !== null ? `${left}/${right}` : null;
  if (pairKey !== lastPairRef.current) {
    lastPairRef.current = pairKey;
    // Inline reset (don't trigger a re-render via useEffect — the
    // existing render is the right place to drop the stale state).
    if (narrative !== null && narrative.left_symbol + "/" + narrative.right_symbol !== pairKey) {
      setNarrative(null);
      setError(null);
    }
  }

  const trigger = useCallback(
    async (opts?: { mode?: NarrativeMode; fresh?: boolean }): Promise<void> => {
      if (left === null || right === null || left === right) return;
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        if (opts?.mode) params.set("mode", opts.mode);
        if (opts?.fresh) params.set("fresh", "true");
        const query = params.toString();
        const url = `/api/compare/${encodeURIComponent(left)}/${encodeURIComponent(right)}/narrative${
          query ? `?${query}` : ""
        }`;
        const res = await fetch(url, { credentials: "same-origin" });
        if (!res.ok) {
          let detail: string;
          try {
            const body = await res.json();
            detail =
              typeof body?.detail === "string" ? body.detail : res.statusText;
          } catch {
            detail = res.statusText;
          }
          throw new Error(`${res.status} ${detail}`);
        }
        const data = (await res.json()) as ComparisonNarrative;
        if (reqRef.current === reqId) {
          setNarrative(data);
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
    [left, right],
  );

  const reset = useCallback((): void => {
    setNarrative(null);
    setError(null);
  }, []);

  return { narrative, loading, error, trigger, reset };
}

/** Iterate sections in the canonical render order. */
export function narrativeSections(
  narrative: ComparisonNarrative,
): Array<{ key: string; section: NarrativeSection }> {
  return [
    { key: "headline", section: narrative.headline },
    { key: "momentum", section: narrative.momentum },
    { key: "valuation", section: narrative.valuation },
    { key: "risk", section: narrative.risk },
    { key: "quality", section: narrative.quality },
    { key: "bottom_line", section: narrative.bottom_line },
  ];
}
