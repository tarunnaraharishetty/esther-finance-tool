/**
 * Key Drivers API client + hook.
 *
 * Mirrors the wire shape produced by `/api/movement/{symbol}/drivers`
 * and the underlying `src/intelligence/movement_drivers.py` dataclasses.
 *
 * The hook auto-fetches on symbol change — the panel is the *first*
 * thing a trader sees on the analyzer page, so it should populate
 * eagerly. ``refresh({fresh: true})`` rebuilds the underlying
 * analyzer + sector reports.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type DriverKind =
  | "technical"
  | "sentiment"
  | "news"
  | "sector"
  | "trust"
  | "calibration";

export type DriverTone = "bull" | "bear" | "warn" | "neutral";

export interface Driver {
  kind: DriverKind;
  label: string;
  detail: string;
  tone: DriverTone;
  salience: number; // [0, 1]
  citations: string[];
}

export interface DriverSources {
  analyzer: boolean;
  sector: boolean;
  snapshot_row: boolean;
}

export interface KeyDriversResponse {
  symbol: string;
  drivers: Driver[];
  considered: number;
  suppressed: number;
  sources: DriverSources;
  cache: "hit" | "miss";
}

interface UseKeyDrivers {
  drivers: KeyDriversResponse | null;
  loading: boolean;
  error: string | null;
  refresh: (opts?: { fresh?: boolean }) => Promise<void>;
}

/**
 * Auto-fetching Key Drivers hook. Re-fetches when the symbol changes.
 */
export function useKeyDrivers(symbol: string | null): UseKeyDrivers {
  const [drivers, setDrivers] = useState<KeyDriversResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<number>(0);

  const fetchFor = useCallback(
    async (sym: string, opts?: { fresh?: boolean }): Promise<void> => {
      const reqId = ++reqRef.current;
      setLoading(true);
      setError(null);
      try {
        const url = `/api/movement/${encodeURIComponent(sym)}/drivers${
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
        const data = (await res.json()) as KeyDriversResponse;
        if (reqRef.current === reqId) {
          setDrivers(data);
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
      setDrivers(null);
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

  return { drivers, loading, error, refresh };
}

/** Tone class mapping shared with other components for visual parity. */
export function toneClasses(tone: DriverTone): {
  border: string;
  bg: string;
  text: string;
  bar: string;
} {
  switch (tone) {
    case "bull":
      return {
        border: "border-bull/40",
        bg: "bg-bull/10",
        text: "text-bull",
        bar: "bg-bull",
      };
    case "bear":
      return {
        border: "border-bear/40",
        bg: "bg-bear/10",
        text: "text-bear",
        bar: "bg-bear",
      };
    case "warn":
      return {
        border: "border-warn/40",
        bg: "bg-warn/10",
        text: "text-warn",
        bar: "bg-warn",
      };
    case "neutral":
      return {
        border: "border-border/50",
        bg: "bg-card/40",
        text: "text-foreground",
        bar: "bg-muted-foreground/60",
      };
  }
}
