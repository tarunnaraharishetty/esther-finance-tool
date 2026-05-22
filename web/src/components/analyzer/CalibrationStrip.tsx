import type { CalibrationReading } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  calibrations: CalibrationReading[];
}

/**
 * Per-pairing calibrated-probability panel.
 *
 * One row per ``(score, outcome)`` pairing the analyzer publishes
 * against. Published readings show hit rate + Wilson CI + observation
 * count; under-sampled readings show "calibration pending" with the
 * raw score value. The contract:
 *
 *  * **Never publish a probability without observation count.**
 *    ``bucket_published === false`` ⇒ no percentage rendered.
 *  * **"Closed lower" / "closed higher" language**, not "chance of".
 *    The hit rate is descriptive — past behavior of similar setups —
 *    not predictive.
 *
 * Returns ``null`` when ``calibrations`` is empty. The wire contract
 * always emits an array, so empty means calibration is disabled on
 * the deployment; hiding the panel is the right state (vs rendering
 * a confusing empty card).
 */
export function CalibrationStrip({ calibrations }: Props) {
  if (calibrations.length === 0) return null;

  const horizonDays = calibrations[0].horizon_days;

  return (
    <section className="surface p-5" data-testid="calibration-strip">
      <header className="mb-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Calibrated Probabilities · {horizonDays}d horizon
        </div>
        <h3 className="mt-1 text-base font-semibold tracking-tight">
          Historical hit rates for similar setups
        </h3>
        <p className="mt-1 font-mono text-[10px] leading-relaxed text-muted-foreground">
          Pooled across watchlist symbols. Past behavior — not a forecast.
        </p>
      </header>

      <ul className="space-y-2">
        {calibrations.map((reading) => (
          <li key={`${reading.score_name}/${reading.outcome_name}`}>
            <CalibrationRow reading={reading} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function CalibrationRow({ reading }: { reading: CalibrationReading }) {
  const display = formatScoreName(reading.score_name);
  const closedDirection = directionLabel(reading.outcome_name);

  if (!reading.bucket_published) {
    return (
      <div
        data-testid={`calibration-row-${reading.score_name}`}
        data-status="pending"
        className="flex flex-wrap items-baseline justify-between gap-2 rounded-md border border-border/40 bg-card/40 px-3 py-2"
      >
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-sm font-semibold text-foreground">
            {display}
          </span>
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            = {Math.round(reading.score_value)}
          </span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          calibration pending
        </span>
      </div>
    );
  }

  // Published path: bucket fields are non-null by contract.
  const hitRate = reading.hit_rate ?? 0;
  const ciLow = reading.confidence_low ?? 0;
  const ciHigh = reading.confidence_high ?? 1;
  const n = reading.n_observations ?? 0;
  const tone = hitRateTone(hitRate, reading.outcome_name);

  return (
    <div
      data-testid={`calibration-row-${reading.score_name}`}
      data-status="published"
      className="flex flex-wrap items-baseline justify-between gap-3 rounded-md border border-border/40 bg-card/40 px-3 py-2"
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className="font-mono text-sm font-semibold text-foreground">
          {display}
        </span>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          = {Math.round(reading.score_value)}
        </span>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          → {(hitRate * 100).toFixed(0)}% {closedDirection}
        </span>
      </div>
      <div className="flex items-baseline gap-2 font-mono text-[10px] tabular-nums">
        <span className={cn("uppercase tracking-wider", tone)}>
          CI {(ciLow * 100).toFixed(0)}–{(ciHigh * 100).toFixed(0)}%
        </span>
        <span className="text-muted-foreground">{n} obs</span>
      </div>
    </div>
  );
}

function formatScoreName(name: string): string {
  // Human-readable label for each pairing. The wire shape uses
  // snake_case; the UI shows trader-facing language ("pullback risk").
  switch (name) {
    case "pullback_risk":
      return "Pullback risk";
    case "rebound_potential":
      return "Rebound potential";
    case "overbought_score":
      return "Overbought";
    case "oversold_score":
      return "Oversold";
    default:
      return name.replace(/_/g, " ");
  }
}

function directionLabel(outcomeName: string): string {
  // ``outcome_name`` is the binary outcome the score is graded against.
  // The label here is past tense: "closed lower" / "closed higher" —
  // not "chance of falling". Descriptive, not predictive.
  switch (outcomeName) {
    case "return_negative":
      return "closed lower";
    case "return_positive":
      return "closed higher";
    default:
      return outcomeName;
  }
}

function hitRateTone(hit: number, outcomeName: string): string {
  // Color the CI label with the same threat-hierarchy as the rest of
  // the app: a high return_negative hit rate is a warning; a high
  // return_positive hit rate is bullish. Below 50% (random-walk) is
  // muted — the bucket isn't telling us much.
  if (hit < 0.5) return "text-muted-foreground";
  if (outcomeName === "return_negative") return "text-bear";
  if (outcomeName === "return_positive") return "text-bull";
  return "text-muted-foreground";
}
