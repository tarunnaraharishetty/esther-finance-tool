/**
 * Compare-two-stocks API client + React hook.
 *
 * Mirrors the wire shape produced by `src/api/compare.py` and
 * `src/intelligence/comparison.py` one-to-one. Strict tsc enforces
 * that any backend schema drift fails the compile rather than
 * crashes mid-render.
 *
 * Verdict semantics
 * -----------------
 * Every metric carries a ``direction`` field that says whether
 * higher or lower values are better. The UI uses this to render the
 * winner chip in the right tone — it never re-derives the verdict
 * locally. That keeps the comparison contract single-sourced in
 * Python.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { AnalyzerReport } from "@/lib/analyzer";

export type CompareWinner = "left" | "right" | "tie" | "n/a";
export type CompareDirection = "higher_better" | "lower_better";

export interface MetricComparison {
  metric: string;
  label: string;
  left_value: number | null;
  right_value: number | null;
  winner: CompareWinner;
  direction: CompareDirection;
  detail: string;
  percent: boolean;
}

export interface ComparisonSection {
  title: string;
  metrics: MetricComparison[];
}

export interface HeadlineMetric {
  label: string;
  left_display: string;
  right_display: string;
  winner: CompareWinner;
}

export interface ComparisonView {
  left_symbol: string;
  right_symbol: string;
  headline: HeadlineMetric[];
  sections: ComparisonSection[];
  overall_winner: CompareWinner;
  warnings: string[];
  /** Raw analyzer report for the left side (or null if it failed to assemble). */
  left_report: AnalyzerReport | null;
  /** Raw analyzer report for the right side (or null if it failed to assemble). */
  right_report: AnalyzerReport | null;
}

interface UseCompareView {
  view: ComparisonView | null;
  loading: boolean;
  error: string | null;
  refresh: (opts?: { fresh?: boolean }) => Promise<void>;
}

/**
 * Fetch the side-by-side comparison for two symbols.
 *
 * Re-fetches whenever either symbol changes. ``refresh({ fresh: true })``
 * tells the backend to bypass the analyzer cache for both sides.
 * Returns ``view = null`` while either symbol is null — the UI
 * renders a placeholder in that case.
 */
export function useCompareView(
  left: string | null,
  right: string | null,
): UseCompareView {
  const [view, setView] = useState<ComparisonView | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);

  const fetchFor = useCallback(
    async (
      l: string,
      r: string,
      opts?: { fresh?: boolean },
    ): Promise<void> => {
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const url = `/api/compare/${encodeURIComponent(l)}/${encodeURIComponent(r)}${
          opts?.fresh ? "?fresh=true" : ""
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
        const data = (await res.json()) as ComparisonView;
        if (reqRef.current === reqId) {
          setView(data);
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
    if (left === null || right === null || left === right) {
      setView(null);
      return;
    }
    void fetchFor(left, right);
  }, [left, right, fetchFor]);

  const refresh = useCallback(
    async (opts?: { fresh?: boolean }): Promise<void> => {
      if (left === null || right === null || left === right) return;
      await fetchFor(left, right, opts);
    },
    [left, right, fetchFor],
  );

  return { view, loading, error, refresh };
}

/**
 * Tone mapping for the winner chip. ``"bull"`` is the side that wins;
 * ``"warn"`` is a tie; ``"muted"`` means "no comparison was possible
 * on this metric" — render the value greyed out.
 */
export function winnerTone(
  side: "left" | "right",
  winner: CompareWinner,
): "bull" | "bear" | "warn" | "muted" {
  if (winner === "n/a") return "muted";
  if (winner === "tie") return "warn";
  if (winner === side) return "bull";
  return "bear";
}

/** Format a numeric metric value with optional percent decoration. */
export function formatMetricValue(
  value: number | null,
  percent: boolean,
): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (percent) {
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(1)}%`;
  }
  // Trust score / 100-scale metrics: round to integer.
  if (Math.abs(value) >= 10) return value.toFixed(0);
  // Provider confidence (0..1) and similar: two decimals.
  return value.toFixed(2);
}
