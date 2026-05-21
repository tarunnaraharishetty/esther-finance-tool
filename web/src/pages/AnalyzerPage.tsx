import { useEffect, useMemo, useState } from "react";
import { Search, Star } from "lucide-react";
import { AnalyzerTab } from "@/components/analyzer/AnalyzerTab";
import { Panel } from "@/components/layout/Panel";
import { Input } from "@/components/ui/input";
import { actionTone, fmtPrice, tierLabel } from "@/lib/format";
import { lookupTicker } from "@/lib/tickers";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
}

/**
 * Financial Analyzer page. Left rail is the symbol picker (same shape
 * as the Research page); the main column hosts the AnalyzerTab fed by
 * `/api/analyzer/{symbol}`.
 */
export function AnalyzerPage({
  snapshot,
  activeSymbol,
  setActiveSymbol,
}: Props) {
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (
      activeSymbol === null ||
      !snapshot.rows.some((r) => r.symbol === activeSymbol)
    ) {
      if (snapshot.rows.length > 0) setActiveSymbol(snapshot.rows[0].symbol);
    }
  }, [activeSymbol, snapshot.rows, setActiveSymbol]);

  const filteredWatchlist = useMemo(() => {
    const q = filter.trim().toUpperCase();
    if (!q) return snapshot.rows;
    return snapshot.rows.filter((r) => r.symbol.includes(q));
  }, [snapshot.rows, filter]);

  return (
    <div className="grid gap-4 lg:grid-cols-12">
      <Panel
        className="lg:col-span-3"
        title="Symbols"
        subtitle={`${filteredWatchlist.length} on watchlist`}
      >
        <div className="space-y-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter watchlist…"
              className="h-8 pl-8 text-xs"
            />
          </div>
          <SymbolList
            rows={filteredWatchlist}
            active={activeSymbol}
            onSelect={setActiveSymbol}
          />
        </div>
      </Panel>

      <div className="space-y-4 lg:col-span-9">
        <AnalyzerTab symbol={activeSymbol} />
      </div>
    </div>
  );
}

function SymbolList({
  rows,
  active,
  onSelect,
}: {
  rows: RecommendationRow[];
  active: string | null;
  onSelect: (s: string) => void;
}) {
  if (rows.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border/40 bg-card/30 px-3 py-4 text-center font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        No matches.
      </div>
    );
  }
  return (
    <ul className="space-y-1">
      {rows.map((r) => (
        <li key={r.symbol}>
          <SymbolButton
            row={r}
            active={r.symbol === active}
            onSelect={onSelect}
          />
        </li>
      ))}
    </ul>
  );
}

function SymbolButton({
  row,
  active,
  onSelect,
}: {
  row: RecommendationRow;
  active: boolean;
  onSelect: (s: string) => void;
}) {
  const tone = actionTone(row.action);
  const ticker = lookupTicker(row.symbol);
  return (
    <button
      type="button"
      onClick={() => onSelect(row.symbol)}
      aria-pressed={active}
      className={cn(
        "relative flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left transition-all duration-150",
        active
          ? "border-primary/50 bg-primary/10 shadow-glow"
          : "border-transparent hover:-translate-y-px hover:border-border/60 hover:bg-card/60",
      )}
    >
      {active && (
        <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-r-full bg-primary" />
      )}
      <Star
        className={cn(
          "h-3 w-3 shrink-0",
          active ? "fill-primary text-primary" : "text-muted-foreground/50",
        )}
      />
      <div className="min-w-0 flex-1">
        <div className="font-mono text-sm font-semibold">{row.symbol}</div>
        {ticker && (
          <div className="truncate font-mono text-[10px] text-muted-foreground">
            {ticker.name}
          </div>
        )}
      </div>
      <div className="text-right">
        <div className="font-mono text-[11px] tabular-nums text-muted-foreground">
          ${fmtPrice(row.last_price)}
        </div>
        <div
          className={cn(
            "font-mono text-[9px] uppercase tracking-wider",
            tone === "bull" && "text-bull",
            tone === "bear" && "text-bear",
            tone === "warn" && "text-warn",
          )}
        >
          {tierLabel(row.tier)}
        </div>
      </div>
    </button>
  );
}
