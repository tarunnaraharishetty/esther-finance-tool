// React hook wrapping the SSE stream from `GET /api/stream`.
//
// Behavior:
// - Opens an EventSource on mount, closes on unmount.
// - Updates `snapshot` on every `data:` event with the parsed JSON.
// - Tracks connection status: `connecting` / `open` / `error`.
// - EventSource itself auto-reconnects on transient drops; we just
//   reflect the current readyState so the UI can paint a status chip.

import { useEffect, useRef, useState } from "react";
import type { DashboardSnapshot } from "./types";

export type StreamStatus = "connecting" | "open" | "error";

export interface UseSnapshotStream {
  snapshot: DashboardSnapshot | null;
  status: StreamStatus;
  lastEventAt: number | null; // performance.now() of the last data event
}

/**
 * Subscribe to `GET /api/stream` for the lifetime of the calling component.
 *
 * The default URL is `/api/stream` (relative) so the Vite dev-server
 * proxy can rewrite it to the FastAPI backend, and the same path works
 * unchanged when the bundle is hosted from the same origin in production.
 *
 * `url` is overridable for tests / staging.
 */
export function useSnapshotStream(url: string = "/api/stream"): UseSnapshotStream {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const source = new EventSource(url);
    sourceRef.current = source;

    source.onopen = () => {
      setStatus("open");
    };

    source.onmessage = (ev: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(ev.data) as DashboardSnapshot;
        setSnapshot(parsed);
        setLastEventAt(performance.now());
      } catch (err) {
        // Malformed payloads are a server bug, not a client one.
        // Log and ignore — the next valid tick will overwrite.
        console.error("snapshot parse failed", err, ev.data);
      }
    };

    source.onerror = () => {
      // EventSource's readyState is the source of truth. CLOSED means
      // we have given up; CONNECTING means we're retrying.
      if (source.readyState === EventSource.CLOSED) {
        setStatus("error");
      } else {
        setStatus("connecting");
      }
    };

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [url]);

  return { snapshot, status, lastEventAt };
}
