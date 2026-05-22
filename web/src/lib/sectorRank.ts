/**
 * Sector-relative ranking API client + hook.
 *
 * Mirrors the wire shape produced by `/api/sector/{symbol}/rank` and
 * the underlying `src/intelligence/sector_ranker.py` dataclasses.
 *
 * Two payload variants:
 *
 *  * ``available=true`` — the cohort had 2+ peers, ``metrics`` carries
 *    the per-metric rank rows.
 *  * ``available=false`` — singleton cohort (no peers) or unknown
 *    sector. ``reason`` is a short human-readable string the UI can
 *    show instead of an empty table.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type RankDirection = "higher_better" | "lower_better";

export interface SectorMetricRank {
  metric: string;
  label: string;
  value: number | null;
  rank: number | null;
  cohort_size: number;
  percentile: number | null; // [0, 100]; 100 = best on this metric
  direction: RankDirection;
  percent: boolean;
}

interface SectorRankBase {
  symbol: string;
  sector: string | null;
  cache: "hit" | "miss";
}

export interface SectorRankAvailable extends SectorRankBase {
  available: true;
  cohort_size: number;
  cohort_symbols: string[];
  metrics: SectorMetricRank[];
}

export interface SectorRankUnavailable extends SectorRankBase {
  available: false;
  cohort_size: number;
  cohort_symbols: string[];
  metrics: SectorMetricRank[];
  reason: string;
}

export type SectorRankResponse = SectorRankAvailable | SectorRankUnavailable;

interface UseSectorRank {
  rank: SectorRankResponse | null;
  loading: boolean;
  error: string | null;
  refresh: (opts?: { fresh?: boolean }) => Promise<void>;
}

/**
 * Auto-fetch the sector ranking for a symbol.
 *
 * Re-fetches when the symbol changes. ``refresh({ fresh: true })``
 * tells the backend to rebuild the cohort from scratch (bypasses both
 * the sector cache and the underlying analyzer caches).
 */
export function useSectorRank(symbol: string | null): UseSectorRank {
  const [rank, setRank] = useState<SectorRankResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);

  const fetchFor = useCallback(
    async (sym: string, opts?: { fresh?: boolean }): Promise<void> => {
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const url = `/api/sector/${encodeURIComponent(sym)}/rank${
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
        const data = (await res.json()) as SectorRankResponse;
        if (reqRef.current === reqId) {
          setRank(data);
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
      setRank(null);
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

  return { rank, loading, error, refresh };
}

/**
 * Map a percentile to one of the app's tonal buckets. Higher
 * percentile = better, so 80+ is bull, 40-80 is muted, below 40 is
 * warn. Null (no peers) collapses to muted.
 */
export function percentileTone(
  pct: number | null,
): "bull" | "warn" | "muted" {
  if (pct === null || !Number.isFinite(pct)) return "muted";
  if (pct >= 80) return "bull";
  if (pct >= 40) return "muted";
  return "warn";
}

/** Format a rank value as "#3 of 24" (or "—" when missing). */
export function formatRank(
  rank: number | null,
  cohortSize: number,
): string {
  if (rank === null || cohortSize <= 0) return "—";
  return `#${rank} of ${cohortSize}`;
}

/** Format a percentile as "87th percentile" (or "—" when missing). */
export function formatPercentile(pct: number | null): string {
  if (pct === null || !Number.isFinite(pct)) return "—";
  const rounded = Math.round(pct);
  const suffix =
    rounded % 100 >= 11 && rounded % 100 <= 13
      ? "th"
      : rounded % 10 === 1
        ? "st"
        : rounded % 10 === 2
          ? "nd"
          : rounded % 10 === 3
            ? "rd"
            : "th";
  return `${rounded}${suffix} percentile`;
}
