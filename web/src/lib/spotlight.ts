/**
 * Watchlist Driver Spotlight API client + hook.
 *
 * Mirrors the wire shape produced by `/api/spotlight` and the
 * underlying `src/intelligence/driver_spotlight.py` dataclasses.
 * Each spotlight entry carries a full :type:`Driver` re-exported
 * from the movement-drivers module so the same rendering atoms
 * (tone classes, kind icons) work in both surfaces.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { Driver } from "@/lib/movementDrivers";

export interface SpotlightEntry {
  rank: number;
  symbol: string;
  driver: Driver;
}

export interface SpotlightResponse {
  entries: SpotlightEntry[];
  considered_symbols: number;
  surfaced_symbols: number;
  watchlist_size: number;
  assembled_symbols: number;
  reason?: string;
  cache: "hit" | "miss";
}

interface UseSpotlight {
  spotlight: SpotlightResponse | null;
  loading: boolean;
  error: string | null;
  refresh: (opts?: { fresh?: boolean }) => Promise<void>;
}

/**
 * Auto-fetching spotlight hook. Refetches on mount; re-runs when
 * ``trigger`` changes (use it to bind the fetch to a watchlist
 * mutation, e.g. add/remove symbol).
 */
export function useSpotlight(trigger?: unknown): UseSpotlight {
  const [spotlight, setSpotlight] = useState<SpotlightResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);

  const fetchOnce = useCallback(
    async (opts?: { fresh?: boolean }): Promise<void> => {
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const url = `/api/spotlight${opts?.fresh ? "?fresh=true" : ""}`;
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
        const data = (await res.json()) as SpotlightResponse;
        if (reqRef.current === reqId) {
          setSpotlight(data);
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
    void fetchOnce();
    // ``trigger`` is intentionally part of the dep array — caller
    // can pass watchlist length / tick to force a refetch on change.
  }, [fetchOnce, trigger]);

  const refresh = useCallback(
    async (opts?: { fresh?: boolean }): Promise<void> => {
      await fetchOnce(opts);
    },
    [fetchOnce],
  );

  return { spotlight, loading, error, refresh };
}
