import type { MarketPulse, PulseEvolution } from "../lib/types";

interface Props {
  pulse: MarketPulse | null;
  evolution: PulseEvolution | null;
}

/**
 * Pulse panel — mirrors the Textual WatchlistHeader's PULSE / REGIME /
 * PATTERNS block. Categorical labels surface verbatim (no recoloring
 * of "bullish" / "bearish" so the trader is reading the same words
 * the engine produced).
 */
export function PulseCard({ pulse, evolution }: Props) {
  if (!pulse) {
    return (
      <section className="panel">
        <h2 className="panel__title">Pulse</h2>
        <div className="panel__body panel__body--empty">no pulse yet</div>
      </section>
    );
  }

  const sentimentClass =
    pulse.sentiment === "bullish"
      ? "pulse__pill pulse__pill--bull"
      : pulse.sentiment === "bearish"
        ? "pulse__pill pulse__pill--bear"
        : "pulse__pill";

  const regimeClass = evolution
    ? `pulse__regime pulse__regime--${evolution.regime.replace(/[^a-z]/gi, "")}`
    : "pulse__regime";

  return (
    <section className="panel">
      <h2 className="panel__title">Pulse</h2>
      <div className="panel__body">
        <div className="pulse__row">
          <span className={sentimentClass}>{pulse.sentiment}</span>
          <span className="pulse__pill">{pulse.conviction}</span>
          <span className="pulse__pill">{pulse.activity}</span>
        </div>
        <p className="pulse__summary">{pulse.summary}</p>
        <dl className="pulse__metrics">
          <dt>bull / bear</dt>
          <dd>
            {pulse.bullish_count} / {pulse.bearish_count}
          </dd>
          <dt>momentum breadth</dt>
          <dd>{(pulse.momentum_breadth * 100).toFixed(0)}%</dd>
          <dt>sentiment breadth</dt>
          <dd>{(pulse.sentiment_breadth * 100).toFixed(0)}%</dd>
          <dt>alert intensity</dt>
          <dd>{pulse.alert_intensity}</dd>
        </dl>
        {evolution && evolution.regime !== "indeterminate" && (
          <div className="pulse__evolution">
            <div className={regimeClass}>regime · {evolution.regime}</div>
            {evolution.patterns.length > 0 && (
              <ul className="pulse__patterns">
                {evolution.patterns.map((p) => (
                  <li key={p.name} title={p.detail}>
                    {p.label}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* STRONG signals — only renders when the 5-tier promoter
            actually fires STRONG_BUY / STRONG_SELL. We do not promote
            anything client-side; an empty list means no symbol cleared
            the promoter's gates this tick, and the section is omitted
            entirely (matches the TUI's omit-when-empty semantics and
            keeps the panel quiet most of the time — the natural
            anti-spam behavior). */}
        {pulse.strongest_symbols.length > 0 && (
          <div className="pulse__strong" data-testid="pulse-strong">
            <div className="detail__label">strong signals</div>
            <ul className="pulse__strong-list">
              {pulse.strongest_symbols.map(([symbol, display]) => {
                const direction = display.includes("BUY") ? "buy" : "sell";
                return (
                  <li
                    key={symbol}
                    className={`pulse__strong-chip pulse__strong-chip--${direction}`}
                  >
                    <span className="pulse__strong-tier">{display}</span>
                    <span className="pulse__strong-sym">{symbol}</span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}
