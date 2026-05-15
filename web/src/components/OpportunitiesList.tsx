import type { DashboardSnapshot } from "../lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  onSelect: (symbol: string) => void;
}

/**
 * Opportunities panel — renders the controller's pre-computed
 * ``ranked_opportunities`` list. Composite-score ordering is set
 * server-side so the web frontend sees the same rank the TUI does
 * (no client-side re-ranking, no drift).
 *
 * Tenure chip (NEW / Nx) is overlaid from ``opp_history`` when
 * available. The first ~3 rationale phrases per row are shown
 * verbatim — those are pre-grounded by the strategy layer and form
 * the trader-facing "why is this an opportunity" line.
 */
export function OpportunitiesList({ snapshot, onSelect }: Props) {
  const ranked = snapshot.ranked_opportunities;

  return (
    <section className="panel">
      <h2 className="panel__title">Opportunities</h2>
      <div className="panel__body">
        {ranked.length === 0 ? (
          <div className="panel__body--empty">no symbols qualify this tick</div>
        ) : (
          <ol className="opps">
            {ranked.map((opp, idx) => {
              const history = snapshot.opp_history[opp.symbol];
              const tenure = history
                ? history.streak === 1
                  ? "NEW"
                  : `${history.streak}x`
                : null;
              return (
                <li
                  key={opp.symbol}
                  className="opps__row"
                  onClick={() => onSelect(opp.symbol)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(opp.symbol);
                    }
                  }}
                  tabIndex={0}
                >
                  <span className="opps__rank">#{idx + 1}</span>
                  <span className="opps__sym">{opp.symbol}</span>
                  <span className={`tier tier--${opp.tier}`}>
                    {opp.tier.replace("_", " ")}
                  </span>
                  <span className="opps__score">
                    {opp.composite_score.toFixed(2)}
                  </span>
                  {tenure && (
                    <span
                      className={`opps__tenure opps__tenure--${
                        tenure === "NEW" ? "new" : "carry"
                      }`}
                    >
                      {tenure}
                    </span>
                  )}
                  {opp.rationale.length > 0 && (
                    <span className="opps__rationale">
                      {opp.rationale.slice(0, 2).join(" · ")}
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
