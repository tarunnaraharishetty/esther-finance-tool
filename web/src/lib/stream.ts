// React hook wrapping the SSE stream from `GET /api/stream`.
//
// Resilience contract:
//   - Auto-reconnect with exponential backoff on connection errors
//     (1s, 2s, 4s, 8s, 16s, capped at 30s). EventSource's built-in
//     reconnect is too aggressive on flaky networks; we manage the
//     loop ourselves.
//   - Backoff resets to zero on a successful `open` event.
//   - Stale-data detection: if no tick arrives within
//     `STALE_THRESHOLD_MS`, the hook surfaces `isStale: true` so the
//     UI can warn the trader before the connection itself fails.
//   - Public state is read-only React state; callers don't need to
//     know about the EventSource lifecycle.

import { useEffect, useRef, useState } from "react";
import type { DashboardSnapshot } from "./types";

export type StreamStatus = "connecting" | "open" | "error";

export interface UseSnapshotStream {
  snapshot: DashboardSnapshot | null;
  status: StreamStatus;
  isStale: boolean;
  /** Wall-clock ms since the last data event arrived, or null if none has. */
  msSinceLastEvent: number | null;
}

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30_000;

// A snapshot is "stale" if the most recent event is older than this.
// The default refresh cadence is 5s server-side; 12s gives ample
// headroom for a couple missed ticks before we cry wolf.
const STALE_THRESHOLD_MS = 12_000;

// How often the hook re-evaluates staleness. Cheap timer, lets the
// React tree show the chip without depending on a fresh event.
const STALE_TICK_MS = 1000;

/**
 * Subscribe to `GET /api/stream` for the lifetime of the calling component.
 *
 * The default URL is `/api/stream` (relative) so the same path works
 * in dev (proxied through Vite to the backend) and in production
 * (same-origin, served from FastAPI).
 */
export function useSnapshotStream(url: string = "/api/stream"): UseSnapshotStream {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState<number>(() => Date.now());

  // Refs so the reconnect loop survives renders without re-subscribing.
  const sourceRef = useRef<EventSource | null>(null);
  const retryTimerRef = useRef<number | null>(null);
  const retryCountRef = useRef<number>(0);
  // Guards against late onerror callbacks queuing a reconnect after
  // the component unmounts.
  const disposedRef = useRef<boolean>(false);

  useEffect(() => {
    disposedRef.current = false;

    const connect = (): void => {
      if (disposedRef.current) return;
      setStatus(retryCountRef.current === 0 ? "connecting" : "connecting");

      const source = new EventSource(url);
      sourceRef.current = source;

      source.onopen = () => {
        if (disposedRef.current) return;
        setStatus("open");
        retryCountRef.current = 0;
      };

      source.onmessage = (ev: MessageEvent<string>) => {
        if (disposedRef.current) return;
        try {
          const parsed = JSON.parse(ev.data) as DashboardSnapshot;
          setSnapshot(parsed);
          setLastEventAt(Date.now());
        } catch (err) {
          // Malformed payload is a server bug, not a client one. Log
          // and skip — the next valid tick overwrites everything.
          console.error("snapshot parse failed", err, ev.data);
        }
      };

      source.onerror = () => {
        // EventSource fires `error` both for transient drops and for
        // hard failures. We close it and own the reconnect schedule
        // ourselves so the backoff is predictable.
        source.close();
        sourceRef.current = null;
        if (disposedRef.current) return;
        setStatus("error");
        const delay = Math.min(
          MAX_BACKOFF_MS,
          INITIAL_BACKOFF_MS * 2 ** retryCountRef.current,
        );
        retryCountRef.current += 1;
        retryTimerRef.current = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposedRef.current = true;
      if (retryTimerRef.current !== null) {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      if (sourceRef.current !== null) {
        sourceRef.current.close();
        sourceRef.current = null;
      }
    };
  }, [url]);

  // Staleness ticker: bump `nowMs` once a second so the derived
  // `isStale` recomputes even when no event has arrived. Cheap;
  // unsubscribed on unmount.
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), STALE_TICK_MS);
    return () => window.clearInterval(id);
  }, []);

  const msSinceLastEvent = lastEventAt === null ? null : nowMs - lastEventAt;
  const isStale =
    status === "open" &&
    msSinceLastEvent !== null &&
    msSinceLastEvent > STALE_THRESHOLD_MS;

  return { snapshot, status, isStale, msSinceLastEvent };
}
