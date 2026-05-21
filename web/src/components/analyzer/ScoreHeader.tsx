import { fmtScore } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  symbol: string;
  lastPrice: number | null;
  overall: number | null;
  technical: number | null;
  fundamental: number | null;
  valuation: number | null;
}

/**
 * Four-ring header — overall + technical + fundamental + valuation.
 * Each ring is an SVG arc that fills proportionally to a [0, 100]
 * score. Color tone gradients from muted -> bull -> warn so the eye
 * picks up "high score" without reading the number.
 */
export function ScoreHeader({
  symbol,
  lastPrice,
  overall,
  technical,
  fundamental,
  valuation,
}: Props) {
  return (
    <section className="surface-premium p-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Financial Analyzer
          </div>
          <h2 className="font-display text-2xl font-semibold tracking-tight">
            {symbol}
          </h2>
          <div className="mt-1 font-mono text-xs text-muted-foreground tabular-nums">
            {lastPrice !== null ? `Last $${lastPrice.toFixed(2)}` : "Last —"}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-5">
          <ScoreRing
            label="Overall"
            score={overall}
            size="lg"
            accent="primary"
          />
          <ScoreRing label="Technical" score={technical} />
          <ScoreRing label="Fundamental" score={fundamental} />
          <ScoreRing label="Valuation" score={valuation} />
        </div>
      </div>
    </section>
  );
}

function ScoreRing({
  label,
  score,
  size = "md",
  accent = "default",
}: {
  label: string;
  score: number | null;
  size?: "md" | "lg";
  accent?: "default" | "primary";
}) {
  const dim = size === "lg" ? "h-20 w-20" : "h-16 w-16";
  const value = score ?? 0;
  const radius = 32;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const tone = scoreColorClass(score, accent);
  return (
    <div className="flex flex-col items-center gap-1">
      <div className={cn("relative grid place-items-center", dim)}>
        <svg viewBox="0 0 72 72" className="h-full w-full -rotate-90">
          <circle
            cx="36"
            cy="36"
            r={radius}
            stroke="hsl(var(--border))"
            strokeOpacity="0.4"
            strokeWidth="5"
            fill="none"
          />
          <circle
            cx="36"
            cy="36"
            r={radius}
            stroke="currentColor"
            strokeWidth="5"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={score === null ? circumference : offset}
            className={cn(tone, "transition-[stroke-dashoffset]")}
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span
            className={cn(
              "font-mono font-semibold tabular-nums",
              size === "lg" ? "text-base" : "text-sm",
              tone,
            )}
          >
            {fmtScore(score)}
          </span>
          <span className="font-mono text-[8px] uppercase tracking-wider text-muted-foreground">
            /100
          </span>
        </div>
      </div>
      <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

function scoreColorClass(
  score: number | null,
  accent: "default" | "primary",
): string {
  if (score === null) return "text-muted-foreground";
  if (accent === "primary") return "text-primary";
  if (score >= 70) return "text-warn";
  if (score >= 40) return "text-bull";
  return "text-muted-foreground";
}
