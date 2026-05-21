import type { TechnicalScores } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  technicals: TechnicalScores | null;
}

/**
 * Horizontal "oversold ←→ overbought" spectrum. The bar shows the
 * difference between the two combiners (oversold subtracted from
 * overbought) mapped onto [-100, +100], with a marker for the
 * confidence-weighted position.
 *
 * Below the bar we surface the seven sub-scores as small chips so the
 * read is auditable in one glance.
 */
export function OverboughtSpectrum({ technicals }: Props) {
  if (technicals === null) {
    return <EmptyState />;
  }
  const ob = technicals.overbought_score ?? 0;
  const os = technicals.oversold_score ?? 0;
  const tilt = clamp(ob - os, -100, 100); // -100 strongly oversold, +100 strongly overbought
  const marker = ((tilt + 100) / 200) * 100; // 0..100% across the bar
  const bias =
    tilt >= 20 ? "Stretched UP" : tilt <= -20 ? "Stretched DOWN" : "Balanced";
  const biasTone =
    tilt >= 20 ? "text-warn" : tilt <= -20 ? "text-bear" : "text-muted-foreground";

  const chips = subscoreChips(technicals);

  return (
    <section className="surface p-5">
      <header className="mb-4 flex items-baseline justify-between">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Overbought · Oversold Spectrum
          </div>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            Posture
          </h3>
        </div>
        <span className={cn("font-mono text-xs uppercase tracking-wider", biasTone)}>
          {bias}
        </span>
      </header>

      <div className="relative h-3 w-full rounded-full bg-gradient-to-r from-bear/40 via-muted/40 to-warn/40">
        <div className="absolute inset-y-0 left-1/2 w-px bg-border/80" />
        <div
          className="absolute -top-1.5 h-6 w-1.5 rounded-full bg-foreground shadow-[0_0_0_4px_hsl(var(--background))]"
          style={{ left: `calc(${marker}% - 3px)` }}
          aria-label={`Position ${Math.round(marker)}% from oversold`}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>Oversold {Math.round(os)}</span>
        <span>Neutral</span>
        <span>Overbought {Math.round(ob)}</span>
      </div>

      <div className="mt-4 flex flex-wrap gap-1.5">
        {chips.map((chip) => (
          <SubscoreChip key={chip.label} chip={chip} />
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryStat
          label="Pullback risk"
          value={technicals.pullback_risk}
          tone="warn"
        />
        <SummaryStat
          label="Rebound potential"
          value={technicals.rebound_potential}
          tone="bull"
        />
        <SummaryStat
          label="RSI"
          value={technicals.raw_rsi}
          format="raw"
          tone="muted"
        />
        <SummaryStat
          label="Tech confidence"
          value={technicals.confidence_score}
          tone="muted"
        />
      </div>
    </section>
  );
}

interface Chip {
  label: string;
  score: number | null;
}

function subscoreChips(t: TechnicalScores): Chip[] {
  return [
    { label: "RSI OB", score: t.subscores.rsi_overbought_score },
    { label: "RSI OS", score: t.subscores.rsi_oversold_score },
    { label: "Bollinger", score: t.subscores.bollinger_extension_score },
    { label: "MA ext", score: t.subscores.ma_extension_score },
    { label: "Vol spike", score: t.subscores.volume_spike_score },
    { label: "ATR", score: t.subscores.atr_volatility_score },
    { label: "Exhaustion", score: t.subscores.momentum_exhaustion_score },
  ];
}

function SubscoreChip({ chip }: { chip: Chip }) {
  const s = chip.score;
  const tone =
    s === null
      ? "border-border/40 bg-card/30 text-muted-foreground/60"
      : s >= 70
        ? "border-warn/40 bg-warn/10 text-warn"
        : s >= 40
          ? "border-bull/40 bg-bull/10 text-bull"
          : "border-border/50 bg-card/60 text-muted-foreground";
  return (
    <span
      className={cn(
        "rounded-md border px-2 py-1 font-mono text-[10px] uppercase tracking-wider tabular-nums",
        tone,
      )}
    >
      {chip.label}{" "}
      <span className="font-semibold">
        {s === null ? "—" : Math.round(s)}
      </span>
    </span>
  );
}

function SummaryStat({
  label,
  value,
  tone,
  format = "score",
}: {
  label: string;
  value: number | null;
  tone: "bull" | "bear" | "warn" | "muted";
  format?: "score" | "raw";
}) {
  const toneClass =
    tone === "bull"
      ? "text-bull"
      : tone === "bear"
        ? "text-bear"
        : tone === "warn"
          ? "text-warn"
          : "text-foreground";
  const display =
    value === null || !Number.isFinite(value)
      ? "—"
      : format === "raw"
        ? value.toFixed(1)
        : Math.round(value).toString();
  return (
    <div className="rounded-md border border-border/40 bg-card/40 px-3 py-2">
      <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className={cn("mt-1 font-mono text-sm tabular-nums", toneClass)}>
        {display}
      </div>
    </div>
  );
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function EmptyState() {
  return (
    <section className="surface p-5">
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        Overbought · Oversold Spectrum
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        Technical scoring unavailable — no recent bars to score.
      </p>
    </section>
  );
}
