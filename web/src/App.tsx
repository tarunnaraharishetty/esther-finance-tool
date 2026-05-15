import { useEffect, useState } from "react";
import { useSnapshotStream } from "./lib/stream";
import { ConnectionIndicator } from "./components/ConnectionIndicator";
import { WatchlistTable } from "./components/WatchlistTable";
import { PulseCard } from "./components/PulseCard";
import { OpportunitiesList } from "./components/OpportunitiesList";
import { AlertsFeed } from "./components/AlertsFeed";
import { SymbolDetail } from "./components/SymbolDetail";

/**
 * Top-level layout. Information-dense by design — Esther's brand is
 * trader workstation, not consumer app.
 *
 *  ┌─ header (connection chip) ───────────────────────────────────┐
 *  ├─ stale banner (only when isStale) ───────────────────────────┤
 *  ├─ watchlist (left, full height) ─┬─ detail (right top) ──────┤
 *  │                                  ├─ pulse + opps (right mid)─┤
 *  │                                  ├─ alerts (right bottom) ───┤
 *  └──────────────────────────────────┴───────────────────────────┘
 */
export default function App() {
  const { snapshot, status, isStale, msSinceLastEvent } = useSnapshotStream();
  const [activeSymbol, setActiveSymbol] = useState<string | null>(null);

  // When a snapshot first arrives, default the active symbol to the
  // first row so the detail panel never sits idle if data is
  // available. After that, the user's selection drives.
  useEffect(() => {
    if (activeSymbol !== null) return;
    if (snapshot && snapshot.rows.length > 0) {
      setActiveSymbol(snapshot.rows[0].symbol);
    }
  }, [snapshot, activeSymbol]);

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__title">
          <span className="app__brand">Esther</span>
          <span className="app__subtitle">decision-support, not execution</span>
        </div>
        <ConnectionIndicator
          status={status}
          isStale={isStale}
          tick={snapshot?.tick ?? null}
          rowCount={snapshot?.rows.length ?? 0}
        />
      </header>

      {/* Stale-data banner — only renders when the connection is
          still open but no event has arrived in a while. Surfaces
          the situation where the trader could otherwise be staring
          at a frozen snapshot and not realize it. */}
      {isStale && msSinceLastEvent !== null && (
        <div className="banner banner--stale" role="status">
          Data may be stale — last update {Math.round(msSinceLastEvent / 1000)}s ago.
          Stream is open but no fresh tick has arrived.
        </div>
      )}

      <main className="app__main">
        {snapshot === null ? (
          <div className="placeholder">waiting for first tick…</div>
        ) : (
          <div className="grid">
            <div className="grid__watchlist">
              <WatchlistTable
                rows={snapshot.rows}
                activeSymbol={activeSymbol}
                onSelect={setActiveSymbol}
              />
            </div>
            <div className="grid__detail">
              <SymbolDetail snapshot={snapshot} symbol={activeSymbol} />
            </div>
            <div className="grid__pulse">
              <PulseCard pulse={snapshot.pulse} evolution={snapshot.pulse_evolution} />
            </div>
            <div className="grid__opps">
              <OpportunitiesList snapshot={snapshot} onSelect={setActiveSymbol} />
            </div>
            <div className="grid__alerts">
              <AlertsFeed
                alerts={snapshot.alerts}
                recentAlerts={snapshot.recent_alerts}
                onSelect={setActiveSymbol}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
