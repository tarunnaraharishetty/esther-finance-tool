/**
 * Per-symbol historical signal outcomes client.
 *
 * Mirrors ``src.intelligence.historical_outcomes.outcomes_to_wire``
 * one-to-one. The analyzer page reads from this to drill down from
 * the pooled :class:`CalibrationReading` into the symbol's own
 * track record. ``HistoricalOutcomes`` is always returned (never
 * absent) — when calibration is disabled at the deployment level,
 * the endpoint 404s and ``error`` carries the reason; the hook never
 * renders fabricated data on absence.
 */

import { useCallback, useEffect, useState } from "react";

/**
 * One bucket of observed outcomes for a single symbol.
 *
 * The publish flag is computed server-side against a configurable
 * ``min_observations`` threshold. ``confidence_low`` / ``confidence_high``
 * are Wilson 95% bounds — wider than the pooled view's because
 * per-symbol N is smaller. Trust framing: never render the hit_rate
 * percentage in headline copy when ``bucket_published`` is false.
 */
export interface HistoricalBucket {
  score_name: string;
  outcome_name: string;
  horizon_days: number;
  bucket_lo: number;
  bucket_hi: number;
  n_observations: number;
  n_hits: number;
  hit_rate: number;
  confidence_low: number;
  confidence_high: number;
  bucket_published: boolean;
}

export interface HistoricalOutcomes {
  symbol: string;
  horizon_days: number;
  total_observations: number;
  settled_observations: number;
  first_observed_at: string | null;
  last_settled_at: string | null;
  buckets: HistoricalBucket[];
}

interface UseHistoricalOutcomes {
  outcomes: HistoricalOutcomes | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * Fetch + cache the per-symbol outcomes for one symbol.
 *
 * 404 ⇒ ``error = "Calibration is disabled on this deployment."`` and
 * ``outcomes = null``. Other errors surface verbatim. Re-fetches when
 * the symbol changes.
 */
export function useHistoricalOutcomes(
  symbol: string | null,
): UseHistoricalOutcomes {
  const [outcomes, setOutcomes] = useState<HistoricalOutcomes | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFor = useCallback(async (sym: string): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/history/${encodeURIComponent(sym)}`, {
        credentials: "same-origin",
      });
      if (res.status === 404) {
        setOutcomes(null);
        setError("Calibration is disabled on this deployment.");
        return;
      }
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
      const data = (await res.json()) as HistoricalOutcomes;
      setOutcomes(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (symbol === null) {
      setOutcomes(null);
      return;
    }
    void fetchFor(symbol);
  }, [symbol, fetchFor]);

  const refresh = useCallback(async (): Promise<void> => {
    if (symbol === null) return;
    await fetchFor(symbol);
  }, [symbol, fetchFor]);

  return { outcomes, loading, error, refresh };
}

/**
 * Human label for a score_name — same mapping used by the pooled
 * ``CalibrationStrip``. Keeping it here too so the two views read
 * with consistent vocabulary.
 */
export function formatScoreName(name: string): string {
  switch (name) {
    case "pullback_risk":
      return "Pullback risk";
    case "rebound_potential":
      return "Rebound potential";
    case "overbought_score":
      return "Overbought";
    case "oversold_score":
      return "Oversold";
    default:
      return name.replace(/_/g, " ");
  }
}

/** Past-tense outcome label, matching the pooled view. */
export function formatOutcomeName(name: string): string {
  switch (name) {
    case "return_negative":
      return "closed lower";
    case "return_positive":
      return "closed higher";
    default:
      return name;
  }
}
