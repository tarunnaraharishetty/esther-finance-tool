import { Brain, TrendingDown, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { summarizeReasoning, type Factor } from "@/lib/reasoning";
import { cn } from "@/lib/utils";
import type { RecommendationRow } from "@/lib/types";

interface Props {
  row: RecommendationRow;
  className?: string;
}

/**
 * "Why is this moving?" panel — pairs the bullish and bearish
 * factor stacks side-by-side with weighted bars. The footer
 * surfaces a net read derived from the factor weights. Factors
 * come from `lib/reasoning.ts`; all language is sourced from
 * snapshot fields, never invented.
 */
export function WhyMoving({ row, className }: Props) {
  const { bullish, bearish, net, tilt } = summarizeReasoning(row);

  return (
    <div className={cn("surface p-5", className)}>
      <header className="mb-4 flex items-center gap-2">
        <div className="grid h-7 w-7 place-items-center rounded-md bg-gradient-to-br from-primary to-accent shadow-glow">
          <Brain className="h-3.5 w-3.5 text-primary-foreground" />
        </div>
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          Why is this moving
        </span>
        <Badge
          variant={net === "bullish" ? "bull" : net === "bearish" ? "bear" : "outline"}
          className="ml-auto"
        >
          Net {net}
        </Badge>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        <FactorColumn
          title="Bullish factors"
          icon={TrendingUp}
          tone="bull"
          factors={bullish}
        />
        <FactorColumn
          title="Bearish factors"
          icon={TrendingDown}
          tone="bear"
          factors={bearish}
        />
      </div>

      <TiltMeter tilt={tilt} />
    </div>
  );
}

function FactorColumn({
  title,
  icon: Icon,
  tone,
  factors,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  tone: "bull" | "bear";
  factors: Factor[];
}) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Icon
          className={cn("h-3.5 w-3.5", tone === "bull" ? "text-bull" : "text-bear")}
        />
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
      </div>
      {factors.length === 0 ? (
        <p className="rounded-md border border-dashed border-border/40 bg-card/30 px-3 py-2 text-xs text-muted-foreground">
          Nothing notable on this side right now.
        </p>
      ) : (
        <ul className="space-y-2">
          {factors.map((f) => (
            <li key={f.label}>
              <FactorBar factor={f} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function FactorBar({ factor }: { factor: Factor }) {
  const pct = Math.max(0.08, Math.min(1, factor.weight)) * 100;
  return (
    <div className="rounded-md border border-border/30 bg-card/30 px-3 py-2 transition-colors hover:border-border/60 hover:bg-card/50">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">{factor.label}</span>
        <span
          className={cn(
            "font-mono text-[10px] tabular uppercase tracking-wide",
            factor.tone === "bull" ? "text-bull" : "text-bear",
          )}
        >
          {Math.round(factor.weight * 100)}
        </span>
      </div>
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted/60">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            factor.tone === "bull"
              ? "bg-gradient-to-r from-bull/60 to-bull"
              : "bg-gradient-to-r from-bear/60 to-bear",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 font-mono text-[10px] text-muted-foreground">
        {factor.detail}
      </p>
    </div>
  );
}

function TiltMeter({ tilt }: { tilt: number }) {
  // -1 = full bear, 0 = balanced, +1 = full bull. Render as a
  // centered needle on a gradient track.
  const clamped = Math.max(-1, Math.min(1, tilt));
  // Map -1..1 → 0..100 (% position from the left).
  const pos = (clamped + 1) * 50;
  return (
    <div className="mt-5 rounded-lg border border-border/40 bg-card/30 p-3">
      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        <span className="text-bear">Bearish</span>
        <span>Tilt</span>
        <span className="text-bull">Bullish</span>
      </div>
      <div className="relative mt-2 h-2 overflow-hidden rounded-full bg-gradient-to-r from-bear/60 via-warn/40 to-bull/60">
        <div
          aria-hidden
          className="absolute top-1/2 h-4 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground shadow-glow transition-all duration-500"
          style={{ left: `${pos}%` }}
        />
      </div>
    </div>
  );
}
