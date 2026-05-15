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
                  <li key={p}>{p}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
