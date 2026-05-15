import type { RecommendationRow } from "../lib/types";

interface Props {
  rows: RecommendationRow[];
  activeSymbol: string | null;
  onSelect: (symbol: string) => void;
}

const ACTION_CLASS: Record<string, string> = {
  buy: "act act--buy",
  sell: "act act--sell",
  hold: "act act--hold",
};

function fmtSigned(x: number): string {
  if (Number.isNaN(x)) return "  -  ";
  const sign = x >= 0 ? "+" : "";
  return `${sign}${x.toFixed(2)}`;
}

function fmtPrice(x: number): string {
  if (Number.isNaN(x)) return "—";
  return x.toFixed(2);
}

function confidenceBar(c: number): string {
  // 10-cell horizontal bar; mirrors the Textual app's `_confidence_bar`.
  const filled = Math.round(Math.max(0, Math.min(1, c)) * 10);
  return "█".repeat(filled) + "·".repeat(10 - filled);
}

/**
 * Watchlist table — primary navigation surface. Click a row to make
 * it the active symbol; the detail panel reads that selection.
 *
 * Column set follows the TUI's `_COLUMN_DEFS` order, minus the
 * timeframe-toggle columns that are dashboard-specific (BAR, BBAND
 * are still useful here, so they stay).
 */
export function WatchlistTable({ rows, activeSymbol, onSelect }: Props) {
  return (
    <table className="watchlist">
      <thead>
        <tr>
          <th>SYM</th>
          <th>ACTION</th>
          <th className="num">CONF</th>
          <th className="bar">BAR</th>
          <th className="num">TECH</th>
          <th className="num">SENT</th>
          <th className="num">RSI</th>
          <th className="num">MACD</th>
          <th className="num">BBAND</th>
          <th className="num">PRICE</th>
          <th className="num">NEWS</th>
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 ? (
          <tr>
            <td colSpan={11} className="empty">
              watchlist is empty — add symbols via the CLI for now.
            </td>
          </tr>
        ) : (
          rows.map((r) => {
            const isActive = r.symbol === activeSymbol;
            const action = r.error ? "err" : r.action;
            const className = `watchlist__row${isActive ? " watchlist__row--active" : ""}${
              r.error ? " watchlist__row--error" : ""
            }`;
            return (
              <tr
                key={r.symbol}
                className={className}
                onClick={() => onSelect(r.symbol)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(r.symbol);
                  }
                }}
                tabIndex={0}
              >
                <td className="sym">{r.symbol}</td>
                <td className={ACTION_CLASS[action] ?? "act"}>
                  {r.error ? "ERR" : r.action.toUpperCase()}
                </td>
                <td className="num">{r.error ? "—" : r.confidence.toFixed(2)}</td>
                <td className="bar">
                  <code>{confidenceBar(r.error ? 0 : r.confidence)}</code>
                </td>
                <td className={`num ${r.technical_score >= 0 ? "pos" : "neg"}`}>
                  {r.error ? "—" : fmtSigned(r.technical_score)}
                </td>
                <td className={`num ${r.sentiment_score >= 0 ? "pos" : "neg"}`}>
                  {r.error ? "—" : fmtSigned(r.sentiment_score)}
                </td>
                <td className="num">{fmtSigned(r.rsi)}</td>
                <td className="num">{fmtSigned(r.macd)}</td>
                <td className="num">{fmtSigned(r.bollinger)}</td>
                <td className="num">{fmtPrice(r.last_price)}</td>
                <td className="num">{r.error ? "" : r.num_news_articles}</td>
              </tr>
            );
          })
        )}
      </tbody>
    </table>
  );
}
