import type { StreamStatus } from "../lib/stream";

interface Props {
  status: StreamStatus;
  tick: number | null;
  rowCount: number;
}

/**
 * Small status pill in the top-right of the layout. Mirrors the
 * Textual dashboard's `STORE` chip semantically — green when healthy,
 * yellow during reconnect, red on permanent failure.
 */
export function ConnectionIndicator({ status, tick, rowCount }: Props) {
  const label = status === "open" ? "live" : status === "connecting" ? "connecting" : "offline";
  return (
    <div className={`conn conn--${status}`} title={`SSE status: ${status}`}>
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
