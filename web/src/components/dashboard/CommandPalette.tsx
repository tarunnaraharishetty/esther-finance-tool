import { useEffect, useMemo, useState } from "react";
import { Command } from "cmdk";
import {
  Activity,
  FileText,
  Gauge,
  GitCompareArrows,
  Globe,
  LayoutDashboard,
  LineChart,
  Newspaper,
  Search,
  Sparkles,
  Star,
  TrendingUp,
} from "lucide-react";
import type { NavKey } from "@/components/layout/Sidebar";
import { actionTone, fmtPrice } from "@/lib/format";
import { searchTickers } from "@/lib/tickers";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  open: boolean;
  onClose: () => void;
  snapshot: DashboardSnapshot | null;
  onSelectSymbol: (symbol: string) => void;
  onNavigate: (nav: NavKey) => void;
}

const PAGES: Array<{ key: NavKey; label: string; icon: typeof LineChart }> = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "watchlist", label: "Watchlist", icon: Star },
  { key: "movers", label: "Market Movers", icon: TrendingUp },
  { key: "charts", label: "Charts", icon: LineChart },
  { key: "news", label: "News Feed", icon: Newspaper },
  { key: "ai", label: "AI Insights", icon: Sparkles },
  { key: "research", label: "Research Thesis", icon: FileText },
  { key: "analyzer", label: "Financial Analyzer", icon: Gauge },
  { key: "compare", label: "Compare", icon: GitCompareArrows },
];

export function CommandPalette({
  open,
  onClose,
  snapshot,
  onSelectSymbol,
  onNavigate,
}: Props) {
  const [search, setSearch] = useState("");
  useEffect(() => {
    if (!open) setSearch("");
  }, [open]);

  const watchlistSymbols = useMemo(
    () => new Set((snapshot?.rows ?? []).map((r) => r.symbol)),
    [snapshot],
  );

  // Universe results — filtered by the query. We slice down to keep
  // the list manageable; the query is the trader's filter, so showing
  // 15-ish off-watchlist results is plenty.
  const universeMatches = useMemo(() => {
    const q = search.trim();
    return searchTickers(q, 50).filter((t) => !watchlistSymbols.has(t.symbol));
  }, [search, watchlistSymbols]);

  if (!open) return null;

  return (
    <div
      role="presentation"
      onClick={onClose}
      className="fixed inset-0 z-50 grid place-items-start justify-center bg-background/60 px-4 pt-[12vh] backdrop-blur-md animate-fade-in-fast"
    >
      <div
        role="dialog"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
        className="surface-premium w-full max-w-xl overflow-hidden animate-slide-up"
      >
        <Command label="Command Palette" loop className="flex flex-col">
          <div className="flex items-center gap-2 border-b border-border/40 px-4">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Command.Input
              autoFocus
              value={search}
              onValueChange={setSearch}
              placeholder="Search any ticker — NVDA, BTC, SPY, AAPL…"
              className="h-12 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground/60"
            />
            <kbd className="hidden rounded border border-border/60 bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground sm:inline">
              ESC
            </kbd>
          </div>

          <Command.List className="max-h-[60vh] overflow-y-auto p-2">
            <Command.Empty className="grid place-items-center px-4 py-10">
              <Search className="mb-2 h-5 w-5 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No results.</p>
              <p className="text-[11px] text-muted-foreground/70">
                Try a ticker like NVDA or a page name.
              </p>
            </Command.Empty>

            {snapshot && snapshot.rows.length > 0 && (
              <Command.Group
                heading="Watchlist"
                className="mb-2 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted-foreground"
              >
                {snapshot.rows.map((row) => {
                  const tone = actionTone(row.action);
                  return (
                    <Command.Item
                      key={row.symbol}
                      value={`${row.symbol} ${row.action} ${row.tier} watchlist`}
                      onSelect={() => {
                        onSelectSymbol(row.symbol);
                        onClose();
                      }}
                      className="group flex cursor-pointer items-center gap-3 rounded-md px-2 py-2 text-sm transition-colors aria-selected:bg-secondary aria-selected:text-foreground"
                    >
                      <Star className="h-4 w-4 fill-primary text-primary" />
                      <span className="font-mono font-semibold">
                        {row.symbol}
                      </span>
                      <span className="ml-auto flex items-center gap-2">
                        <span className="font-mono text-xs tabular text-muted-foreground">
                          ${fmtPrice(row.last_price)}
                        </span>
                        <span
                          className={cn(
                            "rounded-full px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                            tone === "bull" && "bg-bull/15 text-bull",
                            tone === "bear" && "bg-bear/15 text-bear",
                            tone === "warn" && "bg-warn/15 text-warn",
                          )}
                        >
                          {row.action}
                        </span>
                      </span>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            )}

            {universeMatches.length > 0 && (
              <Command.Group
                heading="All Stocks"
                className="mb-2 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted-foreground"
              >
                {universeMatches.slice(0, 15).map((t) => (
                  <Command.Item
                    key={t.symbol}
                    value={`${t.symbol} ${t.name} ${t.sector} ${t.exchange}`}
                    onSelect={() => {
                      onSelectSymbol(t.symbol);
                      onClose();
                    }}
                    className="flex cursor-pointer items-center gap-3 rounded-md px-2 py-2 text-sm transition-colors aria-selected:bg-secondary aria-selected:text-foreground"
                  >
                    <Globe className="h-4 w-4 text-muted-foreground" />
                    <span className="font-mono font-semibold">{t.symbol}</span>
                    <span className="truncate text-xs text-muted-foreground">
                      {t.name}
                    </span>
                    <span className="ml-auto font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
                      {t.exchange === "CRYPTO" ? "crypto" : t.exchange.toLowerCase()}
                    </span>
                  </Command.Item>
                ))}
                {universeMatches.length > 15 && (
                  <div className="px-3 pb-2 pt-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60">
                    + {universeMatches.length - 15} more · keep typing to filter
                  </div>
                )}
              </Command.Group>
            )}

            <Command.Group
              heading="Navigate"
              className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted-foreground"
            >
              {PAGES.map((p) => {
                const Icon = p.icon;
                return (
                  <Command.Item
                    key={p.key}
                    value={`go ${p.label} ${p.key}`}
                    onSelect={() => {
                      onNavigate(p.key);
                      onClose();
                    }}
                    className="flex cursor-pointer items-center gap-3 rounded-md px-2 py-2 text-sm transition-colors aria-selected:bg-secondary aria-selected:text-foreground"
                  >
                    <Icon className="h-4 w-4 text-muted-foreground" />
                    <span>Go to {p.label}</span>
                  </Command.Item>
                );
              })}
            </Command.Group>
          </Command.List>

          <div className="flex items-center justify-between border-t border-border/40 bg-card/40 px-3 py-2 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <Activity className="h-3 w-3" />
              {snapshot?.rows.length ?? 0} watchlist · {universeMatches.length}{" "}
              global matches
            </span>
            <span>
              <kbd className="font-mono">⏎</kbd> select
              <kbd className="ml-2 font-mono">ESC</kbd> close
            </span>
          </div>
        </Command>
      </div>
    </div>
  );
}
