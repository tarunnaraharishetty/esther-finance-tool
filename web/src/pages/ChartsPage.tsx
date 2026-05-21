import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { Globe, Search, Star } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Panel } from "@/components/layout/Panel";
import { ConfidenceMeter } from "@/components/dashboard/ConfidenceMeter";
import { WhyMoving } from "@/components/dashboard/WhyMoving";
import { generateSymbolSummary } from "@/lib/aiSummary";
import type { Timeframe } from "@/lib/chartData";
import { actionTone, fmtPrice, tierLabel } from "@/lib/format";
import { lookupTicker, searchTickers } from "@/lib/tickers";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

const PriceChart = lazy(() =>
  import("@/components/dashboard/PriceChart").then((m) => ({
    default: m.PriceChart,
  })),
);

interface Props {
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
}

/**
 * Charts workstation.
 *
 * Two left-rail groups: **Watchlist** (live rows from the snapshot,
 * full AI panels available on the right) and **All Stocks** (curated
 * universe — picking one renders only the TradingView embed since
 * we have no `RecommendationRow` for it).
 */
export function ChartsPage({ snapshot, activeSymbol, setActiveSymbol }: Props) {
  const [timeframe, setTimeframe] = useState<Timeframe>("1M");
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (
      activeSymbol === null ||
      (!snapshot.rows.some((r) => r.symbol === activeSymbol) &&
        !lookupTicker(activeSymbol))
    ) {
      if (snapshot.rows.length > 0) setActiveSymbol(snapshot.rows[0].symbol);
    }
  }, [activeSymbol, snapshot.rows, setActiveSymbol]);

  const row = useMemo(
    () => snapshot.rows.find((r) => r.symbol === activeSymbol) ?? null,
    [snapshot.rows, activeSymbol],
  );
  const universeTicker = useMemo(
    () => (activeSymbol ? lookupTicker(activeSymbol) : undefined),
    [activeSymbol],
  );
  const isOffWatchlist = row === null && universeTicker !== undefined;

  const filteredWatchlist = useMemo(() => {
    const q = filter.trim().toUpperCase();
    if (!q) return snapshot.rows;
    return snapshot.rows.filter((r) => r.symbol.includes(q));
  }, [snapshot.rows, filter]);

  const watchlistSymbols = useMemo(
    () => new Set(snapshot.rows.map((r) => r.symbol)),
    [snapshot.rows],
  );
  const universeMatches = useMemo(
    () =>
      searchTickers(filter, 60).filter((t) => !watchlistSymbols.has(t.symbol)),
    [filter, watchlistSymbols],
  );

  return (
    <div className="grid gap-4 lg:grid-cols-12">
      <Panel
        className="lg:col-span-3"
        title="Symbols"
        subtitle={`${filteredWatchlist.length + universeMatches.length} matches`}
      >
        <div className="space-y-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter any ticker…"
              className="h-8 pl-8 text-xs"
            />
          </div>

          {filteredWatchlist.length > 0 && (
            <SymbolGroup
              title="Watchlist"
              icon={Star}
              count={filteredWatchlist.length}
            >
              {filteredWatchlist.map((r) => (
                <WatchlistRow
                  key={r.symbol}
                  row={r}
                  active={r.symbol === activeSymbol}
                  onSelect={() => setActiveSymbol(r.symbol)}
                />
              ))}
            </SymbolGroup>
          )}

          {universeMatches.length > 0 && (
            <SymbolGroup
              title="All Stocks"
              icon={Globe}
              count={universeMatches.length}
            >
              {universeMatches.slice(0, 40).map((t) => (
                <UniverseRow
                  key={t.symbol}
                  ticker={t}
                  active={t.symbol === activeSymbol}
                  onSelect={() => setActiveSymbol(t.symbol)}
                />
              ))}
              {universeMatches.length > 40 && (
                <li className="px-3 pt-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60">
                  + {universeMatches.length - 40} more · keep filtering
                </li>
              )}
            </SymbolGroup>
          )}
        </div>
      </Panel>

      <div className="space-y-4 lg:col-span-9">
        {isOffWatchlist && universeTicker ? (
          <OffWatchlistView
            symbol={universeTicker.symbol}
            name={universeTicker.name}
            sector={universeTicker.sector}
            timeframe={timeframe}
            onTimeframeChange={setTimeframe}
          />
        ) : row ? (
          <OnWatchlistView
            row={row}
            timeframe={timeframe}
            onTimeframeChange={setTimeframe}
          />
        ) : (
          <Panel title="Chart" subtitle="—">
            <div className="grid h-full place-items-center text-sm text-muted-foreground">
              Pick a symbol from the left.
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}

function SymbolGroup({
  title,
  icon: Icon,
  count,
  children,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        <Icon className="h-3 w-3" />
        {title}
        <span className="ml-auto font-mono text-muted-foreground/60">
          {count}
        </span>
      </div>
      <ul className="space-y-1">{children}</ul>
    </div>
  );
}

function WatchlistRow({
  row,
  active,
  onSelect,
}: {
  row: RecommendationRow;
  active: boolean;
  onSelect: () => void;
}) {
  const tone = actionTone(row.action);
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
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
        <span className="font-mono text-sm font-semibold">{row.symbol}</span>
        <span className="ml-auto font-mono text-[10px] tabular text-muted-foreground">
          ${fmtPrice(row.last_price)}
        </span>
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            tone === "bull" && "bg-bull",
            tone === "bear" && "bg-bear",
            tone === "warn" && "bg-warn",
          )}
        />
      </button>
    </li>
  );
}

function UniverseRow({
  ticker,
  active,
  onSelect,
}: {
  ticker: ReturnType<typeof lookupTicker>;
  active: boolean;
  onSelect: () => void;
}) {
  if (!ticker) return null;
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
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
        <Globe className="h-3 w-3 shrink-0 text-muted-foreground/60" />
        <span className="font-mono text-sm font-semibold">{ticker.symbol}</span>
        <span className="ml-auto truncate font-mono text-[10px] text-muted-foreground">
          {ticker.name}
        </span>
      </button>
    </li>
  );
}

function OnWatchlistView({
  row,
  timeframe,
  onTimeframeChange,
}: {
  row: RecommendationRow;
  timeframe: Timeframe;
  onTimeframeChange: (tf: Timeframe) => void;
}) {
  return (
    <>
      <Panel title={row.symbol} subtitle={tierLabel(row.tier)}>
        <div className="space-y-5">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={actionTone(row.action)}>
              {row.action.toUpperCase()}
            </Badge>
            <Badge variant="outline">{tierLabel(row.tier)}</Badge>
            <Badge variant="outline">{row.signal_quality}</Badge>
            <Badge variant="outline">{row.stability}</Badge>
            <div className="ml-auto min-w-[200px]">
              <ConfidenceMeter
                value={row.confidence}
                tone={actionTone(row.action)}
                label="Confidence"
                size="sm"
                showTicks={false}
              />
            </div>
          </div>

          <Suspense
            fallback={
              <div className="rounded-xl border border-border/40 bg-card/30 p-3">
                <Skeleton className="h-[460px] w-full opacity-60" />
              </div>
            }
          >
            <PriceChart
              symbol={row.symbol}
              action={row.action}
              lastPrice={row.last_price}
              timeframe={timeframe}
              onTimeframeChange={onTimeframeChange}
              height={460}
            />
          </Suspense>

          <div className="surface-premium relative overflow-hidden p-5">
            <div
              aria-hidden
              className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-primary/15 blur-2xl"
            />
            <p className="relative text-sm leading-relaxed">
              {generateSymbolSummary(row)}
            </p>
          </div>
        </div>
      </Panel>

      <WhyMoving row={row} />
    </>
  );
}

function OffWatchlistView({
  symbol,
  name,
  sector,
  timeframe,
  onTimeframeChange,
}: {
  symbol: string;
  name: string;
  sector: string;
  timeframe: Timeframe;
  onTimeframeChange: (tf: Timeframe) => void;
}) {
  return (
    <Panel title={symbol} subtitle={name}>
      <div className="space-y-5">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline">{sector}</Badge>
          <Badge variant="warn">External symbol</Badge>
          <span className="text-[11px] text-muted-foreground">
            Not on the watchlist — chart only. Add to watchlist via the CLI to
            enable AI panels.
          </span>
        </div>

        <Suspense
          fallback={
            <div className="rounded-xl border border-border/40 bg-card/30 p-3">
              <Skeleton className="h-[460px] w-full opacity-60" />
            </div>
          }
        >
          <PriceChart
            symbol={symbol}
            action="hold"
            lastPrice={0}
            timeframe={timeframe}
            onTimeframeChange={onTimeframeChange}
            height={460}
            defaultKind="tradingview"
          />
        </Suspense>
      </div>
    </Panel>
  );
}
