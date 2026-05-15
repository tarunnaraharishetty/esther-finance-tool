import { useSnapshotStream } from "./lib/stream";
import { ConnectionIndicator } from "./components/ConnectionIndicator";

/**
 * Phase-0 frontend shell: proves the wire is live. The full panel
 * grid lands in the next commit.
 *
 * Layout intentionally austere — Esther's product principle is
 * "information density beats visual flourish."
 */
export default function App() {
  const { snapshot, status } = useSnapshotStream();

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__title">
          <span className="app__brand">Esther</span>
          <span className="app__subtitle">decision-support, not execution</span>
        </div>
        <ConnectionIndicator
          status={status}
          tick={snapshot?.tick ?? null}
          rowCount={snapshot?.rows.length ?? 0}
        />
      </header>

      <main className="app__main">
        {snapshot === null ? (
          <div className="placeholder">waiting for first tick…</div>
        ) : (
          <pre className="placeholder placeholder--debug">
            {/* Phase-0 placeholder — full panels land next commit. */}
            tick {snapshot.tick} at {snapshot.timestamp}
            {"\n"}
            rows: {snapshot.rows.map((r) => r.symbol).join(", ")}
            {"\n"}
            regime: {snapshot.pulse_evolution?.regime ?? "(none)"}
            {"\n"}
            alerts: {snapshot.alerts.length}
          </pre>
        )}
      </main>
    </div>
  );
}
