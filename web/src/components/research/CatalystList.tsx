import { CalendarClock } from "lucide-react";
import type { Catalyst } from "@/lib/research";
import { cn } from "@/lib/utils";
import { ProvenanceProse } from "@/components/prose/ProvenanceProse";

interface Props {
  catalysts: Catalyst[];
}

/**
 * Timeline-style list of upcoming catalysts. Each entry is iconified
 * by impact (bullish/bearish/uncertain) so the trader can scan the
 * skyline at a glance.
 */
export function CatalystList({ catalysts }: Props) {
  if (catalysts.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border/40 bg-card/30 px-3 py-2 text-xs text-muted-foreground">
        No scheduled catalysts on file — connect an earnings calendar provider
        to populate.
      </div>
    );
  }
  return (
    <ul className="space-y-2">
      {catalysts.map((c, i) => (
        <li key={`${c.label}-${i}`}>
          <Row catalyst={c} />
        </li>
      ))}
    </ul>
  );
}

function Row({ catalyst }: { catalyst: Catalyst }) {
  const tone = impactTone(catalyst.impact);
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-md border bg-card/40 px-3 py-2",
        tone === "bull" && "border-bull/30",
        tone === "bear" && "border-bear/30",
        tone === "warn" && "border-warn/30",
      )}
    >
      <div
        className={cn(
          "grid h-7 w-7 shrink-0 place-items-center rounded-md border",
          tone === "bull" && "border-bull/40 bg-bull/10 text-bull",
          tone === "bear" && "border-bear/40 bg-bear/10 text-bear",
          tone === "warn" && "border-warn/40 bg-warn/10 text-warn",
        )}
      >
        <CalendarClock className="h-3.5 w-3.5" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="text-sm font-semibold">{catalyst.label}</span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {catalyst.when}
          </span>
          <span
            className={cn(
              "ml-auto rounded-sm border px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider",
              tone === "bull" && "border-bull/40 bg-bull/10 text-bull",
              tone === "bear" && "border-bear/40 bg-bear/10 text-bear",
              tone === "warn" && "border-warn/40 bg-warn/10 text-warn",
            )}
          >
            {catalyst.impact}
          </span>
        </div>
        <p className="mt-1 text-[12px] leading-relaxed text-muted-foreground">
          <ProvenanceProse
            body={catalyst.detail}
            provenance={catalyst.provenance}
          />
        </p>
      </div>
    </div>
  );
}

function impactTone(impact: string): "bull" | "bear" | "warn" {
  if (impact === "bullish") return "bull";
  if (impact === "bearish") return "bear";
  return "warn";
}
