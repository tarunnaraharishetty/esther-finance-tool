import type { DashboardSnapshot, RecommendationRow } from "../lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  symbol: string | null;
}

function fmtSigned(x: number): string {
  if (Number.isNaN(x)) return "—";
  const sign = x >= 0 ? "+" : "";
  return `${sign}${x.toFixed(2)}`;
}

function rowFor(snapshot: DashboardSnapshot, symbol: string | null): RecommendationRow | null {
  if (symbol === null) return null;
  return snapshot.rows.find((r) => r.symbol === symbol) ?? null;
}

/**
 * Symbol detail — equivalent to the Textual DetailPanel for the
 * cursor row. Shows the full reasoning string, recent headlines,
 * quality grade, intraday alignment (when present), and the
 * opportunity tenure if the symbol is in top-N.
 */
export function SymbolDetail({ snapshot, symbol }: Props) {
  const row = rowFor(snapshot, symbol);

  if (row === null) {
    return (
      <section className="panel">
        <h2 className="panel__title">Detail</h2>
        <div className="panel__body panel__body--empty">
          select a symbol from the watchlist or opportunities
        </div>
      </section>
    );
  }

  const oppHistory = snapshot.opp_history[row.symbol];
  const intradayOpp = snapshot.intraday_opp_history[row.symbol];

  return (
    <section className="panel">
      <h2 className="panel__title">
        Detail · {row.symbol}
        <span className={`tier tier--${row.tier}`}>{row.tier.replace("_", " ")}</span>
      </h2>
      <div className="panel__body panel__body--scroll">
        {row.error ? (
          <div className="detail__error">error: {row.error}</div>
        ) : (
          <>
            <div className="detail__grid">
              <div>
                <span className="detail__label">action</span>
                <span className={`act act--${row.action}`}>{row.action.toUpperCase()}</span>
              </div>
              <div>
                <span className="detail__label">confidence</span>
                <span>{row.confidence.toFixed(2)}</span>
              </div>
              <div>
                <span className="detail__label">last price</span>
                <span>{row.last_price.toFixed(2)}</span>
              </div>
              <div>
                <span className="detail__label">quality</span>
                <span>
                  {row.signal_quality} · {row.stability}
                </span>
              </div>
              <div>
                <span className="detail__label">tech</span>
                <span>{fmtSigned(row.technical_score)}</span>
              </div>
              <div>
                <span className="detail__label">sentiment</span>
                <span>{fmtSigned(row.sentiment_score)}</span>
              </div>
              <div>
                <span className="detail__label">RSI · MACD · BBAND</span>
                <span>
                  {fmtSigned(row.rsi)} · {fmtSigned(row.macd)} · {fmtSigned(row.bollinger)}
                </span>
              </div>
              <div>
                <span className="detail__label">news (24h)</span>
                <span>{row.num_news_articles} article(s)</span>
              </div>
            </div>

            <div className="detail__section">
              <div className="detail__label">reasoning</div>
              <p className="detail__reasoning">{row.reasoning}</p>
            </div>

            {row.quality_reasons.length > 0 && (
              <div className="detail__section">
                <div className="detail__label">quality flags</div>
                <ul className="detail__chips">
                  {row.quality_reasons.map((r) => (
                    <li key={r} className="chip">
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {row.intraday && (
              <div className="detail__section">
                <div className="detail__label">intraday read</div>
                <div className="detail__intraday">
                  <span className={`act act--${row.intraday.action}`}>
                    {row.intraday.action.toUpperCase()}
                  </span>
                  <span>conf {row.intraday.confidence.toFixed(2)}</span>
                  <span>tech {fmtSigned(row.intraday.technical_score)}</span>
                </div>
              </div>
            )}

            {(oppHistory || intradayOpp) && (
              <div className="detail__section">
                <div className="detail__label">opportunity tenure</div>
                <div className="detail__opp">
                  {oppHistory && (
                    <span>
                      daily · streak {oppHistory.streak} ·{" "}
                      {oppHistory.appearances}/{oppHistory.window} in window
                    </span>
                  )}
                  {intradayOpp && (
                    <span>
                      intraday · streak {intradayOpp.streak} ·{" "}
                      {intradayOpp.appearances}/{intradayOpp.window} in window
                    </span>
                  )}
                </div>
              </div>
            )}

            {row.headlines.length > 0 && (
              <div className="detail__section">
                <div className="detail__label">recent headlines</div>
                <ul className="detail__headlines">
                  {row.headlines.slice(0, 5).map((h, i) => (
                    <li key={i}>{h}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
