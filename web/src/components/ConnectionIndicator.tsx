import type { StreamStatus } from "../lib/stream";

interface Props {
  status: StreamStatus;
  isStale: boolean;
  tick: number | null;
  rowCount: number;
}

/**
 * Small status pill in the top-right of the layout.
 *
 * Four visible states:
 *   - **live** (green): SSE connected and a fresh tick arrived recently.
 *   - **stale** (yellow): connection is still open, but no event in
 *     the last ~12s. Surfaces transient backend pauses without
 *     dropping the connection state.
 *   - **reconnecting** (yellow, pulsing): EventSource is reopening
 *     after an error; the hook is in its exponential-backoff loop.
 *   - **offline** (red): hard failure that we haven't yet retried
 *     (rarely visible — we transition through it briefly).
 */
export function ConnectionIndicator({ status, isStale, tick, rowCount }: Props) {
  let label: string;
  let modifier: string;
  if (status === "open") {
    if (isStale) {
      label = "stale";
      modifier = "conn--stale";
    } else {
      label = "live";
      modifier = "conn--open";
    }
  } else if (status === "connecting") {
    label = "reconnecting";
    modifier = "conn--connecting";
  } else {
    label = "offline";
    modifier = "conn--error";
  }

  return (
    <div className={`conn ${modifier}`} title={`SSE status: ${status}${isStale ? " · stale" : ""}`}>
      <span className="conn__dot" aria-hidden="true" />
      <span className="conn__label">{label}</span>
      {tick !== null && (
        <span className="conn__meta">
          tick {tick} · {rowCount} {rowCount === 1 ? "row" : "rows"}
        </span>
      )}
    </div>
  );
}
