import type { MetricEntry } from "@/lib/research";
import { cn } from "@/lib/utils";
import { ProvenanceProse } from "./ProvenanceProse";

interface Props {
  metrics: MetricEntry[];
}

/** Tight grid of metric cards. The first row is technicals; future
 * fundamentals slot in below without rearrangement. */
export function MetricGrid({ metrics }: Props) {
  if (metrics.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
      {metrics.map((m) => (
        <MetricCard key={m.label} metric={m} />
      ))}
    </div>
  );
}

function MetricCard({ metric }: { metric: MetricEntry }) {
  return (
    <div
      className={cn(
        "rounded-md border bg-card/40 px-3 py-2 transition-colors hover:border-border/80 hover:bg-card/60",
        metric.tone === "bull" && "border-bull/30",
        metric.tone === "bear" && "border-bear/30",
        metric.tone === "warn" && "border-warn/30",
        metric.tone === null && "border-border/40",
      )}
    >
      <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {metric.label}
      </div>
      <div
        className={cn(
          "font-mono text-sm font-semibold tabular-nums",
          metric.tone === "bull" && "text-bull",
          metric.tone === "bear" && "text-bear",
          metric.tone === "warn" && "text-warn",
        )}
      >
        <ProvenanceProse body={metric.value} provenance={metric.provenance} />
      </div>
      {metric.delta && (
        <div className="font-mono text-[10px] tabular-nums text-muted-foreground">
          {metric.delta}
        </div>
      )}
    </div>
  );
}
