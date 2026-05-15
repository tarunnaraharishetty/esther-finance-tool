import type { DashboardSnapshot } from "../lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  onSelect: (symbol: string) => void;
}

/**
 * Opportunities panel — shows the symbols currently in the top-N
 * opportunity set with their tenure (NEW / Nx). The actual
 * composite-rank ordering is computed server-side at render time in
 * the Textual dashboard; here we derive what we can from
 * `opp_history`. A future API enhancement could pre-compute
 * `ranked_opportunities` on the snapshot so the frontend renders the
 * trader-facing rank explicitly.
 *
 * Click a row to surface that symbol in the detail panel.
 */
export function OpportunitiesList({ snapshot, onSelect }: Props) {
  const entries = Object.entries(snapshot.opp_history);

  // The dashboard sorts opportunities by composite score (descending).
  // Without the composite on the snapshot we sort by tenure (longer
  // streaks are stickier opportunities), then alphabetically for
  // stability across ticks.
  entries.sort(([as, ah], [bs, bh]) => {
    if (ah.streak !== bh.streak) return bh.streak - ah.streak;
    return as.localeCompare(bs);
  });

  // Look up the per-symbol row for tier + confidence to label each
  // line. A symbol can appear in opp_history but not in rows if it
  // was just removed — defensive: skip those.
  const rowFor = (sym: string) => snapshot.rows.find((r) => r.symbol === sym);

  return (
    <section className="panel">
      <h2 className="panel__title">Opportunities</h2>
      <div className="panel__body">
        {entries.length === 0 ? (
          <div className="panel__body--empty">no symbols in top-N this tick</div>
        ) : (
          <ol className="opps">
            {entries.map(([symbol, history], idx) => {
              const row = rowFor(symbol);
              const tenure = history.streak === 1 ? "NEW" : `${history.streak}x`;
              return (
                <li
                  key={symbol}
                  className="opps__row"
                  onClick={() => onSelect(symbol)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(symbol);
                    }
                  }}
                  tabIndex={0}
                >
                  <span className="opps__rank">#{idx + 1}</span>
                  <span className="opps__sym">{symbol}</span>
                  <span className={`opps__tenure opps__tenure--${tenure === "NEW" ? "new" : "carry"}`}>
                    {tenure}
                  </span>
                  {row && (
                    <span className="opps__detail">
                      {row.action} · conf {row.confidence.toFixed(2)} · {row.signal_quality}
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}
