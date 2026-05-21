import { useMemo } from "react";
import { Activity, ArrowDown, ArrowUp, Zap } from "lucide-react";
import { MarketMovers } from "@/components/dashboard/MarketMovers";
import { Panel } from "@/components/layout/Panel";
import { fmtPct } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  setActiveSymbol: (s: string) => void;
}

export function MoversPage({ snapshot, setActiveSymbol }: Props) {
  const stats = useMemo(() => {
    const sorted = [...snapshot.rows].sort(
      (a, b) => b.combined_score - a.combined_score,
    );
    const top = sorted[0] ?? null;
    const bottom = sorted[sorted.length - 1] ?? null;
    const advancers = sorted.filter((r) => r.combined_score > 0).length;
    const decliners = sorted.filter((r) => r.combined_score < 0).length;
    const breadth =
      snapshot.rows.length > 0 ? advancers / snapshot.rows.length : 0;
    return { top, bottom, advancers, decliners, breadth };
  }, [snapshot.rows]);

  return (
    <Panel
      title="Market Movers"
      subtitle={`${snapshot.rows.length} symbols · ${(stats.breadth * 100).toFixed(0)}% advancing`}
    >
      <div className="mb-4 grid gap-2 md:grid-cols-4">
        <SummaryStat
          icon={ArrowUp}
          label="Advancing"
          value={stats.advancers.toString()}
          tone="bull"
        />
        <SummaryStat
          icon={ArrowDown}
          label="Declining"
          value={stats.decliners.toString()}
          tone="bear"
        />
        <SummaryStat
          icon={Zap}
          label="Top"
          value={stats.top ? `${stats.top.symbol} ${fmtPct(stats.top.combined_score)}` : "—"}
          tone="bull"
        />
        <SummaryStat
          icon={Activity}
          label="Bottom"
          value={
            stats.bottom
              ? `${stats.bottom.symbol} ${fmtPct(stats.bottom.combined_score)}`
              : "—"
          }
          tone="bear"
        />
      </div>

      <MarketMovers snapshot={snapshot} onSelect={setActiveSymbol} />
    </Panel>
  );
}

function SummaryStat({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  tone: "bull" | "bear" | "warn";
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2.5 rounded-md border bg-card/40 px-3 py-2",
        tone === "bull" && "border-bull/30",
        tone === "bear" && "border-bear/30",
        tone === "warn" && "border-warn/30",
      )}
    >
      <Icon
        className={cn(
          "h-3.5 w-3.5",
          tone === "bull" && "text-bull",
          tone === "bear" && "text-bear",
          tone === "warn" && "text-warn",
        )}
      />
      <div className="min-w-0">
        <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        <div className="truncate font-mono text-sm font-semibold tabular">
          {value}
        </div>
      </div>
    </div>
  );
}
