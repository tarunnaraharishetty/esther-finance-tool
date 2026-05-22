import { Layers } from "lucide-react";
import {
  formatPercentile,
  formatRank,
  percentileTone,
  useSectorRank,
  type SectorMetricRank,
} from "@/lib/sectorRank";
import { cn } from "@/lib/utils";

interface Props {
  symbol: string | null;
}

/**
 * Sector-relative ranking panel.
 *
 * For each metric the platform tracks, shows where the symbol stands
 * within its sector cohort (rank, cohort size, percentile + a position
 * bar). Renders ``null`` (no chrome) when the cohort is unusable —
 * either there's no other symbol on the watchlist in this sector or
 * the symbol's sector is unknown. We surface the reason inline rather
 * than hiding it silently so the trader understands the absence.
 */
export function SectorRankPanel({ symbol }: Props) {
  const { rank, loading, error } = useSectorRank(symbol);
  if (symbol === null) return null;
  if (error !== null) return null; // analyzer surface already shows errors.
  if (loading && rank === null) return <SkeletonPanel />;
  if (rank === null) return null;

  if (!rank.available) {
    return (
      <section
        data-testid="sector-rank-panel"
        data-state="unavailable"
        className="surface flex items-start gap-3 p-4"
      >
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/15 to-accent/10">
          <Layers className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Sector Rank
            {rank.sector && (
              <span className="ml-2 normal-case tracking-normal text-muted-foreground/70">
                · {rank.sector}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {rank.reason}. Add more watchlist symbols in this sector to unlock
            peer rankings.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      data-testid="sector-rank-panel"
      data-state="ready"
      className="surface overflow-hidden"
    >
      <header className="flex items-center justify-between gap-2 border-b border-border/40 bg-card/30 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Layers className="h-3.5 w-3.5 text-primary" />
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Sector Rank
            {rank.sector && (
              <span className="ml-2 normal-case tracking-normal">
                · {rank.sector}
              </span>
            )}
          </div>
        </div>
        <div
          className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
          title={rank.cohort_symbols.join(", ")}
        >
          cohort · {rank.cohort_size}
        </div>
      </header>
      <div>
        {rank.metrics.map((m) => (
          <RankRow key={m.metric} metric={m} />
        ))}
      </div>
    </section>
  );
}

function RankRow({ metric }: { metric: SectorMetricRank }) {
  const tone = percentileTone(metric.percentile);
  return (
    <div
      data-testid={`sector-rank-row-${metric.metric}`}
      data-rank={metric.rank ?? "none"}
      className="grid grid-cols-[1fr_auto] items-center gap-3 border-t border-border/30 px-4 py-2.5 first:border-t-0"
    >
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="text-[13px] font-medium text-foreground">
            {metric.label}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {metric.direction === "higher_better"
              ? "higher · better"
              : "lower · better"}
          </span>
        </div>
        <PositionBar
          percentile={metric.percentile}
          direction={metric.direction}
        />
      </div>
      <div className="whitespace-nowrap text-right">
        <div
          className={cn(
            "font-mono text-[12px] font-semibold tabular-nums",
            tone === "bull" && "text-bull",
            tone === "warn" && "text-warn",
            tone === "muted" && "text-muted-foreground",
          )}
        >
          {formatRank(metric.rank, metric.cohort_size)}
        </div>
        <div className="font-mono text-[10px] text-muted-foreground">
          {formatPercentile(metric.percentile)}
        </div>
      </div>
    </div>
  );
}

function PositionBar({
  percentile,
  direction,
}: {
  percentile: number | null;
  direction: SectorMetricRank["direction"];
}) {
  if (percentile === null || !Number.isFinite(percentile)) {
    return (
      <div
        data-testid="position-bar"
        data-state="empty"
        className="mt-1.5 h-1.5 rounded-full bg-muted/40"
      />
    );
  }
  const tone = percentileTone(percentile);
  // Clamp to [0, 100] defensively in case backend math drifts.
  const clamped = Math.max(0, Math.min(100, percentile));
  return (
    <div
      data-testid="position-bar"
      data-percentile={Math.round(clamped)}
      data-direction={direction}
      className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted/40"
    >
      <div
        className={cn(
          "h-full rounded-full transition-all",
          tone === "bull" && "bg-bull",
          tone === "warn" && "bg-warn",
          tone === "muted" && "bg-muted-foreground/60",
        )}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

function SkeletonPanel() {
  return (
    <section
      data-testid="sector-rank-panel"
      data-state="loading"
      className="surface p-4"
    >
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        Sector Rank
      </div>
      <div className="mt-3 space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-6 w-full animate-pulse rounded bg-muted/40" />
        ))}
      </div>
    </section>
  );
}
