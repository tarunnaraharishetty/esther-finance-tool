import type { ScenarioModel } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  scenarios: ScenarioModel | null;
  lastPrice: number | null;
}

/**
 * Probabilistic-range panel for the analyzer card.
 *
 * Surfaces what the lognormal scenario model says about the symbol's
 * dispersion over the configured horizon. Three pieces:
 *
 *  * A 20/50/80 quantile bar with the current price marked.
 *  * Tail probabilities ``P(S_T < bear)`` and ``P(S_T > bull)`` when
 *    the valuation ensemble produced bear/bull cases. Hidden
 *    otherwise — we don't fabricate probabilities from absent inputs.
 *  * The model-card caveats verbatim from ``scenarios.notes``.
 *
 * Trust framing — the language here is deliberate: "model-implied
 * probability" never "chance of". The model uses μ=0 by design, so
 * probabilities describe *current dispersion under current vol*, not
 * a directional forecast. Mislabeling these as forecasts would
 * undo the institutional-grade trust contract the platform is built on.
 */
export function ScenarioRange({ scenarios, lastPrice }: Props) {
  if (scenarios === null) {
    return (
      <section className="surface p-5" data-testid="scenario-empty">
        <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Probabilistic Range
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Scenario model unavailable — insufficient bar history or no
          price to anchor on.
        </p>
      </section>
    );
  }

  const { quantile_20: q20, quantile_50: q50, quantile_80: q80 } = scenarios;
  // Lay out the marker for the current price relative to the 20/80
  // band. When ``lastPrice`` sits outside the band we clamp so the
  // marker stays inside the rendered range — the numeric label still
  // shows the actual price so the trader sees the gap.
  const minBound = Math.min(q20, lastPrice ?? q20);
  const maxBound = Math.max(q80, lastPrice ?? q80);
  const range = Math.max(maxBound - minBound, 1e-9);
  const markerPct =
    lastPrice === null
      ? 50
      : Math.max(
          0,
          Math.min(100, ((lastPrice - minBound) / range) * 100),
        );

  const horizonLabel = `${scenarios.horizon_days}d horizon`;
  const volPercent = (scenarios.annualized_vol * 100).toFixed(1);

  return (
    <section className="surface p-5" data-testid="scenario-range">
      <header className="mb-4 flex items-baseline justify-between">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Probabilistic Range · {horizonLabel}
          </div>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            Model-implied dispersion
          </h3>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          σ {volPercent}% ann
        </span>
      </header>

      <div
        className="relative h-3 w-full rounded-full bg-gradient-to-r from-muted/40 via-primary/30 to-muted/40"
        data-testid="scenario-bar"
      >
        {lastPrice !== null && (
          <div
            className="absolute -top-1.5 h-6 w-1.5 rounded-full bg-foreground shadow-[0_0_0_4px_hsl(var(--background))]"
            style={{ left: `calc(${markerPct}% - 3px)` }}
            aria-label={`Last price $${lastPrice.toFixed(2)}`}
          />
        )}
      </div>
      <div className="mt-2 grid grid-cols-3 font-mono text-[10px] tabular-nums">
        <Quantile label="q20" value={q20} align="left" />
        <Quantile label="q50" value={q50} align="center" />
        <Quantile label="q80" value={q80} align="right" />
      </div>

      <TailProbabilities scenarios={scenarios} />

      <ul
        className="mt-4 space-y-1.5 font-mono text-[10px] leading-relaxed text-muted-foreground"
        data-testid="scenario-notes"
      >
        {scenarios.notes.map((note, i) => (
          <li key={i} className="flex items-start gap-1.5">
            <span className="mt-1 inline-block h-1 w-1 shrink-0 rounded-full bg-muted-foreground/50" />
            <span>{note}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Quantile({
  label,
  value,
  align,
}: {
  label: string;
  value: number;
  align: "left" | "center" | "right";
}) {
  return (
    <div
      className={cn(
        align === "left" && "text-left",
        align === "center" && "text-center",
        align === "right" && "text-right",
      )}
    >
      <div className="uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-foreground">${value.toFixed(2)}</div>
    </div>
  );
}

function TailProbabilities({ scenarios }: { scenarios: ScenarioModel }) {
  const below = scenarios.prob_below_bear;
  const above = scenarios.prob_above_bull;
  if (below === null && above === null) return null;
  return (
    <div className="mt-4 flex flex-wrap gap-3" data-testid="scenario-tails">
      {below !== null && (
        <TailChip
          label="P(< bear)"
          value={below}
          tone="bear"
          testid="scenario-tail-below"
        />
      )}
      {above !== null && (
        <TailChip
          label="P(> bull)"
          value={above}
          tone="bull"
          testid="scenario-tail-above"
        />
      )}
    </div>
  );
}

function TailChip({
  label,
  value,
  tone,
  testid,
}: {
  label: string;
  value: number;
  tone: "bull" | "bear";
  testid: string;
}) {
  const toneClass =
    tone === "bull"
      ? "border-bull/40 bg-bull/10 text-bull"
      : "border-bear/40 bg-bear/10 text-bear";
  return (
    <div
      data-testid={testid}
      className={cn(
        "rounded-md border px-3 py-2 font-mono",
        toneClass,
      )}
    >
      <div className="text-[9px] uppercase tracking-wider opacity-80">
        {label}
      </div>
      <div className="mt-0.5 text-sm font-semibold tabular-nums">
        {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}
