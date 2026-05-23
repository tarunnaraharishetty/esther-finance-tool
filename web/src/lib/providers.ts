/**
 * Providers health API client + hook.
 *
 * Mirrors the wire shape produced by `/api/health/providers` after
 * the trust-breakdown + per-field accuracy extension. The ProvidersPage
 * uses this as its sole data source.
 *
 * Auto-refresh
 * ------------
 * The Providers page is a slow-changing surface (trust weights drift
 * over hours, not seconds) so the hook polls every 60s by default.
 * Pass ``refreshIntervalMs: 0`` to disable polling for tests.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface ProviderTrustBreakdown {
  provider: string;
  weight: number; // [0.5, 1.2]
  accuracy: number | null; // [0, 1] or null when below min_events
  uptime: number | null; // [0, 1] or null
  latency_p95_ms: number | null;
  latency_score: number | null; // [0, 1] or null
  events: number;
  health_calls: number;
  cold_start: boolean;
}

export interface ProviderFieldAccuracy {
  provider: string;
  field: string;
  total: number;
  agreed: number;
  accuracy: number; // [0, 1]
}

export interface ProviderFieldAccuracyBlock {
  provider: string;
  window_seconds: number;
  total_events: number;
  total_agreed: number;
  overall_accuracy: number;
  by_field: ProviderFieldAccuracy[];
}

export interface ProviderRow {
  provider: string;
  window_seconds: number;
  total: number;
  ok: number;
  rate_limited: number;
  unavailable: number;
  transient: number;
  empty: number;
  skipped: number;
  success_rate: number;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  last_seen_at: string | null;
  last_error_at: string | null;
  breakdown: Record<string, number>;
  trust_breakdown: ProviderTrustBreakdown | null;
  field_accuracy: ProviderFieldAccuracyBlock | null;
}

export interface RecentFailure {
  provider: string;
  symbol: string;
  status: string;
  latency_ms: number;
  checked_at: string;
  error_message: string | null;
}

export interface ProvidersResponse {
  as_of: string;
  window_seconds: number;
  providers: ProviderRow[];
  recent_failures: RecentFailure[];
}

interface UseProviders {
  data: ProvidersResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/**
 * Auto-polling hook for the providers health endpoint.
 *
 * Pass ``window`` like "1h" / "24h" to control the rolling window.
 * ``refreshIntervalMs`` defaults to 60_000; pass 0 to disable polling.
 */
export function useProviders(
  window: string = "1h",
  refreshIntervalMs: number = 60_000,
): UseProviders {
  const [data, setData] = useState<ProvidersResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);

  const fetchOnce = useCallback(async (): Promise<void> => {
    const reqId = ++reqRef.current;
    setLoading(true);
    setError(null);
    try {
      const url = `/api/health/providers?window=${encodeURIComponent(window)}`;
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
      const body = (await res.json()) as ProvidersResponse;
      if (reqRef.current === reqId) {
        setData(body);
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
  }, [window]);

  useEffect(() => {
    void fetchOnce();
    if (refreshIntervalMs <= 0) return;
    const id = globalThis.setInterval(() => {
      void fetchOnce();
    }, refreshIntervalMs);
    return () => globalThis.clearInterval(id);
  }, [fetchOnce, refreshIntervalMs]);

  return { data, loading, error, refresh: fetchOnce };
}

/**
 * Map a trust weight to a tonal bucket.
 * - Weight ≥ 1.05 = bull (provider exceeds baseline)
 * - Weight in [0.85, 1.05) = neutral
 * - Weight in [0.7, 0.85) = warn
 * - Weight < 0.7 = bear
 */
export function trustTone(weight: number): "bull" | "warn" | "bear" | "muted" {
  if (!Number.isFinite(weight)) return "muted";
  if (weight >= 1.05) return "bull";
  if (weight >= 0.85) return "muted";
  if (weight >= 0.7) return "warn";
  return "bear";
}

/** Format the trust weight (always 3 chars: 0.85 / 1.15). */
export function formatWeight(weight: number): string {
  if (!Number.isFinite(weight)) return "—";
  return weight.toFixed(2);
}

/** Format a [0, 1] rate as a percent ("83%"). */
export function formatPercent(rate: number | null): string {
  if (rate === null || !Number.isFinite(rate)) return "—";
  return `${Math.round(rate * 100)}%`;
}

/** Format latency in ms ("142ms" / "1.2s"). */
export function formatLatency(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}
