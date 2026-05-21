import { ArrowDown, ArrowUp } from "lucide-react";
import { MiniSpark } from "@/components/ui/MiniSpark";
import { actionTone, fmtPct, fmtPrice } from "@/lib/format";
import { sparklineFor } from "@/lib/sparkline";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  onSelect: (symbol: string) => void;
}

/**
 * Top gainers + top losers. Percentage rendered here is
 * `combined_score` (signed conviction in [-1, 1]) presented as a
 * faux-pct until OHLCV ships. The two columns are visually
 * differentiated by gradient header borders to signal direction
 * at a glance.
 */
export function MarketMovers({ snapshot, onSelect }: Props) {
  const sorted = [...snapshot.rows].sort(
    (a, b) => b.combined_score - a.combined_score,
  );
  const gainers = sorted.slice(0, 4);
  const losers = sorted.slice(-4).reverse();

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <MoverColumn
        title="Top Gainers"
        rows={gainers}
        tone="bull"
        onSelect={onSelect}
      />
      <MoverColumn
        title="Top Losers"
        rows={losers}
        tone="bear"
        onSelect={onSelect}
      />
    </div>
  );
}

function MoverColumn({
  title,
  rows,
  tone,
  onSelect,
}: {
  title: string;
  rows: RecommendationRow[];
  tone: "bull" | "bear";
  onSelect: (s: string) => void;
}) {
  const Icon = tone === "bull" ? ArrowUp : ArrowDown;
  const avg =
    rows.length > 0 ? rows.reduce((s, r) => s + r.combined_score, 0) / rows.length : 0;
  return (
    <div className="space-y-2">
      <div
        className={cn(
          "flex items-center gap-2 rounded-md border-l-2 px-2.5 py-1.5",
          tone === "bull"
            ? "border-l-bull bg-bull/5"
            : "border-l-bear bg-bear/5",
        )}
      >
        <Icon
          className={cn(
            "h-3.5 w-3.5",
            tone === "bull" ? "text-bull" : "text-bear",
          )}
        />
        <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {title}
        </span>
        <span
          className={cn(
            "ml-auto font-mono text-[10px] tabular",
            tone === "bull" ? "text-bull" : "text-bear",
          )}
        >
          avg {(avg * 100).toFixed(1)}%
        </span>
      </div>
      <ul className="space-y-1.5">
        {rows.map((row) => (
          <MoverRow key={row.symbol} row={row} onSelect={onSelect} />
        ))}
      </ul>
    </div>
  );
}

function MoverRow({
  row,
  onSelect,
}: {
  row: RecommendationRow;
  onSelect: (s: string) => void;
}) {
  const tone = actionTone(row.action);
  const pct = row.combined_score;
  const PctIcon = pct >= 0 ? ArrowUp : ArrowDown;
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(row.symbol)}
        className="group flex w-full items-center gap-3 rounded-lg border border-border/40 bg-card/40 px-3 py-2 text-left transition-all hover:-translate-y-px hover:border-border hover:bg-card hover:shadow-md"
      >
        <div className="min-w-[3.5rem]">
          <div className="font-mono text-sm font-semibold">{row.symbol}</div>
          <div className="font-mono text-[10px] text-muted-foreground tabular">
            ${fmtPrice(row.last_price)}
          </div>
        </div>
        <div className="hidden flex-1 md:block">
          <MiniSpark
            data={sparklineFor(row.symbol, row.action, row.last_price)}
            tone={tone}
            height={28}
          />
        </div>
        <div className="ml-auto text-right">
          <div
            className={cn(
              "inline-flex items-center gap-0.5 font-mono text-sm font-semibold tabular",
              tone === "bull" && "text-bull",
              tone === "bear" && "text-bear",
              tone === "warn" && "text-warn",
            )}
          >
            <PctIcon className="h-3 w-3" />
            {fmtPct(pct, 1)}
          </div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {row.action}
          </div>
        </div>
      </button>
    </li>
  );
}
