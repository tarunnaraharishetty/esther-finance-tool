import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import {
  AlertOctagon,
  Brain,
  Building2,
  CalendarClock,
  ChartBar,
  CircuitBoard,
  FileText,
  Gauge,
  Globe,
  LineChart,
  Newspaper,
  Search,
  Star,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Panel } from "@/components/layout/Panel";
import { ResearchHeader } from "@/components/research/ResearchHeader";
import {
  ProseAndBullets,
  ThesisSection,
} from "@/components/research/ThesisSection";
import { BullBearStack } from "@/components/research/BullBearStack";
import { OutlookTriad } from "@/components/research/OutlookTriad";
import { MetricGrid } from "@/components/research/MetricGrid";
import { CatalystList } from "@/components/research/CatalystList";
import { useResearchThesis } from "@/lib/research";
import { actionTone, fmtPrice, tierLabel } from "@/lib/format";
import { lookupTicker } from "@/lib/tickers";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

const TradingViewChart = lazy(() =>
  import("@/components/dashboard/TradingViewChart").then((m) => ({
    default: m.TradingViewChart,
  })),
);

interface Props {
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
}

/**
 * Research workstation. Left rail is the symbol picker (mirrors the
 * Charts page pattern); the main column is a stack of institutional-
 * style report sections backed by `/api/research/{symbol}`.
 */
export function ResearchPage({
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

  const { thesis, loading, error, refresh } = useResearchThesis(activeSymbol);

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
        {error && (
          <div className="surface flex items-start gap-3 border-bear/40 p-4">
            <AlertOctagon className="mt-0.5 h-4 w-4 shrink-0 text-bear" />
            <div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-bear">
                Research load failed
              </div>
              <p className="mt-1 text-sm text-muted-foreground">{error}</p>
            </div>
          </div>
        )}

        {!thesis && !error && (loading || activeSymbol !== null) && (
          <ResearchSkeleton />
        )}

        {thesis && (
          <>
            <ResearchHeader
              thesis={thesis}
              onRefresh={() => {
                void refresh({ fresh: true });
              }}
              refreshing={loading}
            />

            <Suspense fallback={<ChartSkeleton />}>
              <ChartPanel
                row={snapshot.rows.find((r) => r.symbol === thesis.symbol) ?? null}
                symbol={thesis.symbol}
              />
            </Suspense>

            <ThesisSection
              eyebrow="Section 01"
              title="Company Overview"
              icon={Building2}
            >
              <ProseAndBullets
                body={thesis.company_overview.body}
                bullets={thesis.company_overview.bullets}
                provenance={thesis.company_overview.provenance}
              />
            </ThesisSection>

            <ThesisSection eyebrow="Section 02" title="Key Metrics" icon={Gauge}>
              <MetricGrid metrics={thesis.metrics} />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 03"
              title="Bull vs Bear Thesis"
              icon={ChartBar}
            >
              <BullBearStack
                bull={thesis.bull_thesis}
                bear={thesis.bear_thesis}
              />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 04"
              title="Technical Analysis"
              icon={LineChart}
            >
              <ProseAndBullets
                body={thesis.technical_analysis.body}
                bullets={thesis.technical_analysis.bullets}
                provenance={thesis.technical_analysis.provenance}
              />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 05"
              title="Fundamental Analysis"
              icon={CircuitBoard}
            >
              <ProseAndBullets
                body={thesis.fundamental_analysis.body}
                bullets={thesis.fundamental_analysis.bullets}
                provenance={thesis.fundamental_analysis.provenance}
              />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 06"
              title="Sentiment & News"
              icon={Newspaper}
            >
              <ProseAndBullets
                body={thesis.sentiment_news.body}
                bullets={thesis.sentiment_news.bullets}
                provenance={thesis.sentiment_news.provenance}
              />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 07"
              title="Catalysts"
              icon={CalendarClock}
            >
              <CatalystList catalysts={thesis.catalysts} />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 08"
              title="Risk Assessment"
              icon={AlertOctagon}
            >
              <ProseAndBullets
                body={thesis.risk_assessment.body}
                bullets={thesis.risk_assessment.bullets}
                provenance={thesis.risk_assessment.provenance}
              />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 09"
              title="AI Investment Outlook"
              icon={Brain}
            >
              <OutlookTriad outlook={thesis.outlook} />
            </ThesisSection>

            <ThesisSection
              eyebrow="Section 10"
              title="Explainability"
              icon={FileText}
            >
              <ProseAndBullets
                body={thesis.explainability.body}
                bullets={thesis.explainability.bullets}
                provenance={thesis.explainability.provenance}
              />
            </ThesisSection>

            <DisclosurePanel thesis={thesis} />
          </>
        )}
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
          <SymbolButton row={r} active={r.symbol === active} onSelect={onSelect} />
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

function ChartPanel({
  row,
  symbol,
}: {
  row: RecommendationRow | null;
  symbol: string;
}) {
  return (
    <section className="surface overflow-hidden">
      <header className="flex items-center gap-2 border-b border-border/40 px-4 py-3">
        <div className="grid h-7 w-7 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <Globe className="h-3.5 w-3.5 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
            Live Tape
          </div>
          <h2 className="font-display text-base font-semibold tracking-tight">
            {symbol} · TradingView
          </h2>
        </div>
        {row && (
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            last ${fmtPrice(row.last_price)}
          </span>
        )}
      </header>
      <div className="h-[460px]">
        <TradingViewChart symbol={symbol} timeframe="1M" height={460} />
      </div>
    </section>
  );
}

function ChartSkeleton() {
  return (
    <div className="surface p-3">
      <Skeleton className="h-[460px] w-full opacity-60" />
    </div>
  );
}

function ResearchSkeleton() {
  return (
    <div className="space-y-4">
      <div className="surface-premium p-5">
        <Skeleton className="h-4 w-32 opacity-60" />
        <Skeleton className="mt-3 h-10 w-40 opacity-60" />
        <Skeleton className="mt-3 h-4 w-2/3 opacity-60" />
      </div>
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="surface p-4">
          <Skeleton className="h-4 w-1/3 opacity-60" />
          <Skeleton className="mt-3 h-20 w-full opacity-60" />
        </div>
      ))}
    </div>
  );
}

function DisclosurePanel({ thesis }: { thesis: ResearchThesisLike }) {
  return (
    <section className="surface p-4">
      <div className="font-mono text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
        Disclosures & Data Sources
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {thesis.data_sources.map((s: string) => (
          <span
            key={s}
            className="rounded-sm border border-border/50 bg-card/40 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
          >
            {s}
          </span>
        ))}
      </div>
      <ul className="mt-3 space-y-1">
        {thesis.disclaimers.map((d: string, i: number) => (
          <li
            key={i}
            className="text-[11px] leading-relaxed text-muted-foreground"
          >
            {d}
          </li>
        ))}
      </ul>
    </section>
  );
}

type ResearchThesisLike = {
  data_sources: string[];
  disclaimers: string[];
};
